# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requires the BF16 Elastic checkpoint and a GPU with at least 80 GiB."""

import os

import pytest

from vllm import SamplingParams


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
