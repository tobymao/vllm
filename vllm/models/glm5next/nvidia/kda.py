# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 KDA modeling adapter."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """Adapt the shared out-buffer KDA layer to GLM's tensor-returning block."""

    enable_b12x_kda_decode = True
    b12x_kda_null_state_index = 0

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections stay BF16. The native FP8 checkpoint lists every one of
        # them under modules_to_not_convert by its HF name (q_proj, k_proj, ...),
        # which the fused in_proj_qkvgfab never matches, and ships no scales for
        # them -- so the quant config must not reach this layer at all. Same guard
        # as upstream's Glm5NextLinearAttention; the NVFP4 (modelopt_mixed) export
        # describes each projection explicitly, which is why it never tripped.
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        super().forward(hidden_states, positions, output)
        return output


__all__ = ["Glm5NextLinearAttention"]
