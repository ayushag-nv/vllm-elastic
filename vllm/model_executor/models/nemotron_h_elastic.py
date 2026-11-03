# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime width slicing with shared attention and Mamba state dimensions."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm import ir
from vllm.config import VllmConfig
from vllm.config.elastic import ElasticConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_grouped_topk,
)
from vllm.model_executor.models.nemotron_h import (
    NemotronHAttention,
    NemotronHMLP,
    NemotronHModel,
    NemotronHMoE,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


class ElasticLinear(nn.Module):
    def __init__(self, layer: nn.Module, state: "NemotronHElasticModel", output: bool):
        super().__init__()
        object.__setattr__(self, "linear", layer)
        self.weight = layer.weight
        self.bias = layer.bias
        self.quant_method = layer.quant_method
        # Do not register the parent model as a child module.
        object.__setattr__(self, "state", state)
        self.output = output

    def load_weights(self, weights):
        if hasattr(self.linear, "load_weights"):
            return self.linear.load_weights(weights)
        return AutoWeightsLoader(self.linear).load_weights(weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        width = self.state.hidden_size if self.output else self.weight.shape[0]
        weight = self.weight[:width, : x.shape[-1]]
        bias = self.bias[:width] if self.bias is not None else None
        return F.linear(x, weight, bias), None


class ElasticNorm(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.variance_epsilon

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        weight = self.weight[: x.shape[-1]]
        if residual is None:
            return ir.ops.rms_norm(x, weight, self.eps)
        return ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, self.eps)


class ElasticMoE(nn.Module):
    def __init__(self, layer: NemotronHMoE, state: "NemotronHElasticModel"):
        super().__init__()
        self.gate = layer.gate
        self.experts = layer.experts
        self.shared_experts = layer.shared_experts
        object.__setattr__(self, "state", state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self.state.config
        hidden = self.state.hidden_size
        logits = F.linear(x.float(), self.gate.weight[:, :hidden].float())
        if self.state.phase == "big":
            return self.experts(hidden_states=x, router_logits=logits)
        weights, indices = fused_grouped_topk(
            x,
            logits,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            self.gate.e_score_correction_bias,
            config.n_group,
            config.topk_group,
            scoring_func="sigmoid",
        )
        intermediate = self.state.elastic.small_intermediate_size
        experts = self.experts.routed_experts
        output = fused_experts(
            x.contiguous(),
            experts.w13_weight[:, :intermediate, :hidden],
            experts.w2_weight[:, :hidden, :intermediate],
            weights,
            indices,
            activation=MoEActivation.from_str(config.mlp_hidden_act).without_mul(),
        )
        output = output * config.routed_scaling_factor
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)
        return output


class NemotronHElasticModel(NemotronHModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        elastic = ElasticConfig.from_vllm_config(vllm_config)
        assert elastic is not None
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.elastic = elastic
        self.phase = "big"
        for layer in self.layers:
            layer.norm = ElasticNorm(layer.norm)
            mixer = layer.mixer
            if isinstance(mixer, NemotronHMoE):
                if mixer.shared_experts is not None:
                    self._slice_mlp(mixer.shared_experts)
                # Keep metadata used by NemotronHForCausalLM's MoE setup.
                replacement = ElasticMoE(mixer, self)
                for name in (
                    "n_logical_experts",
                    "n_physical_experts",
                    "n_local_physical_experts",
                    "n_routed_experts",
                    "n_shared_experts",
                    "n_redundant_experts",
                ):
                    setattr(replacement, name, getattr(mixer, name))
                layer.mixer = replacement
            elif isinstance(mixer, NemotronHMLP):
                self._slice_mlp(mixer)
            elif isinstance(mixer, NemotronHAttention):
                mixer.qkv_proj = ElasticLinear(mixer.qkv_proj, self, False)
                mixer.o_proj = ElasticLinear(mixer.o_proj, self, True)
            else:
                mixer.in_proj = ElasticLinear(mixer.in_proj, self, False)
                mixer.out_proj = ElasticLinear(mixer.out_proj, self, True)
        self.norm_f = ElasticNorm(self.norm_f)

    def _slice_mlp(self, mlp: NemotronHMLP) -> None:
        mlp.up_proj = ElasticLinear(mlp.up_proj, self, False)
        mlp.down_proj = ElasticLinear(mlp.down_proj, self, True)

    @property
    def hidden_size(self) -> int:
        return (
            self.elastic.small_hidden_size
            if self.phase == "small"
            else self.elastic.hidden_size
        )

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        inputs_embeds = inputs_embeds[..., : self.hidden_size].contiguous()
        hidden = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        # The runner and logits head keep a fixed output width across both graphs.
        return F.pad(hidden, (0, self.elastic.hidden_size - self.hidden_size))
