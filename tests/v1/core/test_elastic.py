# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import NemotronHConfig

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    KernelConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.elastic import elastic_phase
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def elastic_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    hf = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=128,
        moe_intermediate_size=128,
        vocab_size=256,
        num_hidden_layers=2,
        hybrid_override_pattern="M*",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        elastic_use_lazy_slicing=True,
        elastic_thinking={
            "elastic_hidden_size": 64,
            "elastic_moe_intermediate_size": 64,
        },
        elastic_think_end_token_id=13,
    )
    hf.save_pretrained(tmp_path)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            max_model_len=256,
            enforce_eager=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=256,
            max_model_len=256,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        kernel_config=KernelConfig(moe_backend="triton"),
        compilation_config=CompilationConfig(mode=0, cudagraph_mode="NONE"),
    )


def request(name, phase):
    return Request(
        name,
        [13, 1, 2],
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": phase}),
        pooling_params=None,
    )


def make_scheduler(config, num_blocks=128):
    config.cache_config.num_gpu_blocks = num_blocks
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    return Scheduler(
        config,
        KVCacheConfig(num_blocks, [], [KVCacheGroupSpec(["layer"], spec)]),
        StructuredOutputManager(config),
        block_size=16,
    )


def step(scheduler, tokens):
    scheduled = scheduler.schedule()
    ids = list(scheduled.num_scheduled_tokens)
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[tokens for _ in ids],
        ),
    )
    return scheduled


def test_auto_switch_uses_generated_tokens_only(elastic_config):
    scheduler = make_scheduler(elastic_config)
    r = request("auto", "auto")
    scheduler.add_request(r)
    assert step(scheduler, [4]).elastic_phase == "small"
    assert step(scheduler, [13]).elastic_phase == "small"
    assert step(scheduler, [13]).elastic_phase == "big"
    assert r.elastic_transition_position == r.num_prompt_tokens + 1
    assert step(scheduler, [4]).elastic_phase == "big"
    assert scheduler.elastic.request_phase(request("fixed", "small")) == "small"


@pytest.mark.parametrize("small_count", [1, 2])
def test_interleaved_cohorts_and_full_capacity(elastic_config, small_count):
    scheduler = make_scheduler(elastic_config)
    requests = [("small-1", "small"), ("big", "big")]
    if small_count == 2:
        requests.append(("small-2", "small"))
    for name, phase in requests:
        scheduler.add_request(request(name, phase))
    first = scheduler.schedule()
    assert first.elastic_phase == "small"
    assert set(first.num_scheduled_tokens) == {
        f"small-{i + 1}" for i in range(small_count)
    }
    ids = list(first.num_scheduled_tokens)
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[4]] * len(ids),
        ),
    )
    # The waiting big request cannot run while both slots are held by small requests.
    second = scheduler.schedule()
    assert second.elastic_phase == ("small" if small_count == 2 else "big")
    assert set(second.num_scheduled_tokens) == (
        set(ids) if small_count == 2 else {"big"}
    )


@pytest.mark.parametrize(
    "request_cls,fields",
    [
        (CompletionRequest, {"prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
def test_api_phase_reaches_sampling_without_mutating_request(request_cls, fields):
    r = request_cls(
        model="elastic", elastic_phase="big", vllm_xargs={"other": 1}, **fields
    )
    params = r.to_sampling_params(10, {})
    assert params.extra_args == {"other": 1, "elastic_phase": "big"}
    assert r.vllm_xargs == {"other": 1}
    with pytest.raises(ValueError):
        request_cls(model="elastic", elastic_phase="invalid", **fields)
    with pytest.raises(ValueError):
        elastic_phase({"elastic_phase": "invalid"})


@pytest.mark.parametrize("phase,enabled", [("bad", True), ("small", False)])
def test_invalid_phase_is_a_client_error(elastic_config, phase, enabled):
    elastic_config.model_config.hf_config.elastic_use_lazy_slicing = enabled
    processor = InputProcessor(elastic_config)
    with pytest.raises(VLLMValidationError, match="elastic_phase"):
        processor.process_inputs(
            "invalid",
            {"type": "token", "prompt_token_ids": [1, 2]},
            SamplingParams(max_tokens=1, extra_args={"elastic_phase": phase}),
            ("generate",),
        )


@pytest.mark.parametrize("prompt_len", [16, 17])
def test_preemption_replays_auto_at_original_widths(elastic_config, prompt_len):
    """Cache pressure must not rebuild the small prefix at big width."""
    blocks_per_prompt = (prompt_len + 15) // 16
    num_blocks = 2 * blocks_per_prompt + 1
    scheduler = make_scheduler(elastic_config, num_blocks)

    older = Request(
        "older",
        [1] * (blocks_per_prompt * 16),
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": "big"}),
        None,
    )
    auto = Request(
        "auto",
        [1] * prompt_len,
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": "auto"}),
        None,
    )
    scheduler.add_request(older)
    step(scheduler, [4])
    scheduler.add_request(auto)
    assert step(scheduler, [13]).elastic_phase == "small"
    step(scheduler, [4])
    assert auto.status == RequestStatus.PREEMPTED
    scheduler.finish_requests([older.request_id], RequestStatus.FINISHED_ABORTED)
    replay = step(scheduler, [])
    assert replay.elastic_phase == "small"
    assert replay.num_scheduled_tokens == {"auto": prompt_len}
    resumed = step(scheduler, [4])
    assert resumed.elastic_phase == "big"
    assert resumed.num_scheduled_tokens == {"auto": 1}
    assert list(auto.output_token_ids) == [13, 4]


@pytest.mark.parametrize(
    "request_cls,fields",
    [
        (CompletionRequest, {"prompt": "hello"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hello"}]}),
    ],
)
def test_beam_search_rejects_explicit_elastic_phase(request_cls, fields):
    request = request_cls(
        model="elastic", use_beam_search=True, elastic_phase="big", **fields
    )
    with pytest.raises(ValueError, match="[Bb]eam search.*Elastic"):
        request.to_beam_search_params(8, {})


def test_resumable_input_is_rejected_for_elastic(elastic_config):
    with pytest.raises(VLLMValidationError, match="resumable input"):
        InputProcessor(elastic_config).process_inputs(
            "session",
            {"type": "token", "prompt_token_ids": [1, 2]},
            SamplingParams(max_tokens=1),
            ("generate",),
            resumable=True,
        )


@pytest.mark.parametrize("prompt_len,evict", [(16, False), (17, False), (16, True)])
def test_resumed_cache_hit_selects_width_at_hit_position(
    elastic_config, prompt_len, evict
):
    elastic_config.cache_config.enable_prefix_caching = True
    scheduler = make_scheduler(elastic_config)
    init_none_hash(sha256)
    auto = Request(
        "auto",
        [1] * prompt_len,
        SamplingParams(max_tokens=8, extra_args={"elastic_phase": "auto"}),
        None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    scheduler.add_request(auto)
    step(scheduler, [13])
    scheduler.running.remove(auto)
    scheduler._preempt_request(auto, auto.arrival_time)
    expected_phase = "big" if prompt_len == 16 and not evict else "small"
    for _ in range(3):
        scheduled = scheduler.schedule()
        if scheduled.num_scheduled_tokens:
            break
        scheduler.update_from_output(scheduled, ModelRunnerOutput([], {}))
        if evict:
            assert scheduler.reset_prefix_cache()
    assert scheduled.elastic_phase == expected_phase
    assert scheduled.num_scheduled_tokens == {"auto": prompt_len if evict else 1}
    assert auto.elastic_cache_hit_tokens == (0 if evict else 16)
