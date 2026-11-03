# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vllm.config.compilation import CompilationMode, CUDAGraphMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

ElasticPhase = Literal["small", "big", "auto"]


def elastic_phase(extra_args: dict | None) -> ElasticPhase:
    phase = (extra_args or {}).get("elastic_phase", "auto")
    if phase not in ("small", "big", "auto"):
        raise ValueError("elastic_phase must be 'small', 'big', or 'auto'")
    return phase


@dataclass
class ElasticConfig:
    hidden_size: int
    intermediate_size: int
    small_hidden_size: int
    small_intermediate_size: int
    think_end_token_id: int

    @classmethod
    def from_vllm_config(cls, config: "VllmConfig") -> "ElasticConfig | None":
        hf = config.model_config.hf_config
        if getattr(hf, "elastic_use_lazy_slicing", False) is not True:
            return None
        small = getattr(hf, "elastic_thinking", {})
        result = cls(
            hf.hidden_size,
            hf.moe_intermediate_size,
            small.get("elastic_hidden_size", 0),
            small.get("elastic_moe_intermediate_size", 0),
            getattr(hf, "elastic_think_end_token_id", -1),
        )
        for child, parent in (
            (result.small_hidden_size, result.hidden_size),
            (result.small_intermediate_size, result.intermediate_size),
        ):
            if not isinstance(child, int) or not 0 < child <= parent or child % 64:
                raise ValueError(
                    "Elastic dimensions must be positive multiples of 64 "
                    "within the full dimensions"
                )
        if not 0 <= result.think_end_token_id < hf.vocab_size:
            raise ValueError("Elastic requires a valid elastic_think_end_token_id")
        parallel = config.parallel_config

        if (
            hf.model_type != "nemotron_h"
            or str(config.model_config.dtype) != "torch.bfloat16"
            or config.quant_config is not None
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
            or config.speculative_config is not None
            or config.lora_config is not None
            or config.kv_transfer_config is not None
            or config.scheduler_config.async_scheduling
            or not config.use_v2_model_runner
            or config.compilation_config.mode != CompilationMode.NONE
            or config.compilation_config.cudagraph_mode
            not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
            or config.kernel_config.moe_backend != "triton"
            or getattr(hf, "moe_latent_size", None) is not None
        ):
            raise ValueError(
                "Elastic requires BF16 Nemotron-H, one GPU, Model Runner V2, "
                "synchronous scheduling, Triton MoE, compilation mode NONE, "
                "and CUDA graph mode NONE or FULL_DECODE_ONLY; "
                "quantization, LoRA, KV connectors and speculative decoding "
                "are not supported"
            )
        return result

    def request_phase(self, request: "Request") -> str:
        params = request.sampling_params
        phase = elastic_phase(params.extra_args if params else None)
        if phase == "auto":
            return (
                "big"
                if self.think_end_token_id in request.output_token_ids
                else "small"
            )
        return phase

    def select_phase(self, requests: Iterable["Request"], previous: str | None) -> str:
        phases = dict.fromkeys(self.request_phase(r) for r in requests)
        return next((p for p in phases if p != previous), next(iter(phases), "big"))
