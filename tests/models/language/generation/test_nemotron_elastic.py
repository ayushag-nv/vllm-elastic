# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny numerical tests and a manual 80-GiB real-checkpoint smoke test."""

import os
from copy import deepcopy

import pytest
import torch
from transformers import NemotronHConfig, NemotronHForCausalLM

from vllm import SamplingParams
from vllm.sampling_params import BeamSearchParams


@pytest.fixture
def tiny_elastic_checkpoint(tmp_path):
    torch.manual_seed(0)
    config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        vocab_size=256,
        num_hidden_layers=3,
        hybrid_override_pattern="M*E",
        mamba_num_heads=4,
        mamba_head_dim=64,
        ssm_state_size=16,
        n_groups=1,
        conv_kernel=4,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_shared_expert_intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        eos_token_id=2,
        bos_token_id=1,
        elastic_use_lazy_slicing=True,
        elastic_thinking={
            "elastic_hidden_size": 64,
            "elastic_moe_intermediate_size": 64,
        },
        elastic_think_end_token_id=13,
    )
    full = NemotronHForCausalLM(config).to(torch.bfloat16).eval()
    # Avoid near-tied expert choices changing between BF16 kernel implementations.
    full.model.layers[2].mixer.gate.e_score_correction_bias.copy_(
        torch.arange(config.n_routed_experts)
    )
    full.save_pretrained(tmp_path)
    small_config = deepcopy(config)
    small_config.hidden_size = 64
    small_config.moe_intermediate_size = 64
    small = NemotronHForCausalLM(small_config).to(torch.bfloat16).eval()
    # Materialize the small reference; inference does not use the Elastic wrappers.
    full_weights = full.state_dict()
    small.load_state_dict(
        {
            name: full_weights[name][
                tuple(slice(0, dim) for dim in tensor.shape)
            ].clone()
            for name, tensor in small.state_dict().items()
        }
    )
    return str(tmp_path), {"big": full, "small": small}


def dense_logprobs(output):
    return torch.tensor(
        [[row[i].logprob for i in range(256)] for row in output.logprobs]
    )


@torch.inference_mode()
def reference_logprobs(models, prompt, trace, phase):
    current = "small" if phase == "auto" else phase
    cache = None
    inputs = torch.tensor([prompt])
    scores = []
    for token in trace:
        result = models[current](inputs, past_key_values=cache, use_cache=True)
        scores.append(result.logits[0, -1].float().log_softmax(-1))
        cache = result.past_key_values
        inputs = torch.tensor([[token]])
        if phase == "auto" and token == 13:
            current = "big"
    return torch.stack(scores)


@pytest.mark.hybrid_model
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_elastic_logits_and_recovery(
    vllm_runner, monkeypatch, tiny_elastic_checkpoint, enforce_eager
):
    """Check logits, including after a real EngineCore cache reset/preemption."""
    path, models = tiny_elastic_checkpoint
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(
        path,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        gpu_memory_utilization=0.08,
        async_scheduling=False,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
        enable_prompt_embeds=True,
        enable_chunked_prefill=True,
        enable_trace_replay=True,
        max_logprobs=256,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2],
        },
    ) as runner:
        llm = runner.llm
        trace = [4, 13, 5, 6]
        for prompt_len in (17, 79, 80, 161):
            prompt = [1] + [42] * (prompt_len - 1)
            for phase in ("small", "big", "auto"):
                params = SamplingParams(
                    temperature=0,
                    max_tokens=len(trace),
                    ignore_eos=True,
                    logprobs=256,
                    trace_decode_token_ids=trace,
                    extra_args={"elastic_phase": phase},
                )
                assert llm.reset_prefix_cache()
                result = llm.generate({"prompt_token_ids": prompt}, params)[0].outputs[
                    0
                ]
                expected = reference_logprobs(models, prompt, trace, phase)
                torch.testing.assert_close(
                    dense_logprobs(result), expected, rtol=0, atol=0.01
                )
                if phase != "auto":
                    continue
                for reset_after in (1, 2, 3):
                    assert llm.reset_prefix_cache()
                    engine = llm.llm_engine
                    engine.add_request("recover", {"prompt_token_ids": prompt}, params)
                    reset = False
                    for _ in range(32):
                        outputs = engine.step()
                        if not outputs:
                            continue
                        output = outputs[0]
                        if (
                            not reset
                            and len(output.outputs[0].token_ids) >= reset_after
                        ):
                            assert llm.reset_prefix_cache(reset_running_requests=True)
                            reset = True
                        if output.finished:
                            break
                    assert output.finished and reset
                    assert output.outputs[0].token_ids == trace
                    torch.testing.assert_close(
                        dense_logprobs(output.outputs[0]), expected, rtol=0, atol=0.01
                    )
        assert llm.reset_prefix_cache()
        embeddings = models["big"].get_input_embeddings()(torch.tensor(prompt)).detach()
        embedded = llm.generate({"prompt_embeds": embeddings}, params)[0].outputs[0]
        torch.testing.assert_close(
            dense_logprobs(embedded), dense_logprobs(result), rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="Beam search.*Elastic"):
            llm.beam_search(
                [{"prompt_token_ids": [1, 4]}],
                BeamSearchParams(beam_width=2, max_tokens=1),
            )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_elastic_switching_and_shared_cache(vllm_runner, monkeypatch, enforce_eager):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Keep model inspection in-process; no callable serialization is needed.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model = os.environ.get(
        "ELASTIC_TEST_MODEL", "nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16"
    )
    with vllm_runner(
        model,
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.88,
        async_scheduling=False,
        enforce_eager=enforce_eager,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
        enable_trace_replay=True,
    ) as runner:
        llm = runner.llm
        marker = llm.apply_model(lambda m: m.config.elastic_think_end_token_id)[0]
        for mode, expected in (("small", "small"), ("big", "big"), ("auto", "big")):
            params = SamplingParams(
                temperature=0,
                max_tokens=3,
                trace_decode_token_ids=[marker, 42, 43],
                extra_args={"elastic_phase": mode},
            )
            output = llm.generate("Complete this sentence:", params)[0]
            assert output.outputs[0].token_ids == [marker, 42, 43]
            assert llm.apply_model(lambda m: m.model.phase) == [expected]
        # Both widths must progress when submitted together.
        params = [
            SamplingParams(
                temperature=0,
                max_tokens=4,
                ignore_eos=True,
                extra_args={"elastic_phase": mode},
            )
            for mode in ("small", "big") * 4
        ]
        outputs = llm.generate([f"Count from {i}:" for i in range(8)], params)
        assert len(outputs) == 8
        assert all(len(o.outputs[0].token_ids) == 4 for o in outputs)
        # The second width reuses the first width's real cache blocks.
        llm.reset_prefix_cache()
        prefix = {"prompt_token_ids": [42] * 5000}
        cached = []
        for mode in ("small", "big", "small"):
            output = llm.generate(
                prefix,
                SamplingParams(
                    temperature=0, max_tokens=1, extra_args={"elastic_phase": mode}
                ),
            )[0]
            cached.append(output.num_cached_tokens)
        assert cached[0] == 0
        assert cached[1] > 0 and cached[2] > 0
