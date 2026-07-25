# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from fractions import Fraction

import torch
from compressed_tensors.quantization import ActivationOrdering

import vllm.envs as envs
from vllm.distributed.utils import verify_group_size_divides_partition
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_fp8_linear,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    2: scalar_types.uint2b2,
    3: scalar_types.uint3b4,
    4: scalar_types.uint4b8,
    5: scalar_types.uint5b16,
    6: scalar_types.uint6b32,
    7: scalar_types.uint7b64,
    8: scalar_types.uint8b128,
}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        symmetric: bool | None = True,
        actorder: ActivationOrdering | None = None,
        layer_name: str | None = None,
    ):
        self.num_bits = num_bits
        self.pack_factor = Fraction(32, num_bits)
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.layer_name = layer_name

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "Pack-quantized format requires group quantization "
                "or channelwise quantization, but found no group "
                "size and strategy is not channelwise."
            )

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {list(WNA16_SUPPORTED_TYPES_MAP)}"
            )

        if not self.symmetric and num_bits not in WNA16_ZP_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Asymmetric quantization not supported for "
                f"num_bits = {num_bits}. Supported: "
                f"{list(WNA16_ZP_SUPPORTED_TYPES_MAP)}"
            )

        self.quant_type = (
            WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
            if not self.symmetric
            else WNA16_SUPPORTED_TYPES_MAP[num_bits]
        )

    @classmethod
    def get_min_capability(cls) -> int:
        # Turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype
        if not hasattr(layer, "has_bias"):
            layer.has_bias = False

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsWNA16", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        if kernel_type is MarlinLinearKernel:
            input_dtype = get_marlin_input_dtype(self.layer_name)
            if input_dtype is not None:
                mp_linear_kernel_config.act_type = input_dtype

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel
        )

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            verify_group_size_divides_partition(
                input_size_per_partition, group_size, self.layer_name
            )
            scales_and_zp_size = input_size_per_partition // group_size

        packed_input_dim = math.ceil(input_size_per_partition * self.num_bits / 32)
        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                packed_input_dim,
                dtype=torch.int32,
            ),
        )

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }

        packed_output_dim = math.ceil(output_size_per_partition * self.num_bits / 32)
        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(
                packed_output_dim,
                scales_and_zp_size,
                dtype=torch.int32,
            ),
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
            if not self.symmetric:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(
                    input_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
            w_gidx_param_name="weight_g_idx",
        )

        # GLM-5.2 fp8-dense: optionally route INT8 dense linears through the
        # blockwise-FP8 DeepGEMM kernel instead of Marlin w8a16 (int8->fp8-block
        # requant happens in process_weights_after_loading). None = keep Marlin.
        self.fp8_dense_kernel = self._maybe_build_fp8_dense_kernel(
            input_size_per_partition, output_size_per_partition, params_dtype
        )

    def _maybe_build_fp8_dense_kernel(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        params_dtype: torch.dtype,
    ):
        # Gated behind VLLM_GLM_FP8_DENSE. Only symmetric 8-bit weight-only
        # (uint8b128) dense linears with no act-order are eligible; the shape must
        # satisfy DeepGEMM's N%64==0 / K%128==0 requirement (this auto-excludes
        # tiny layers such as the MoE router, which stay on Marlin). The INT4 MoE
        # experts never reach this Linear scheme, so they are untouched.
        if not envs.VLLM_GLM_FP8_DENSE:
            return None
        if self.num_bits != 8 or not self.symmetric or self.has_g_idx:
            return None
        if params_dtype != torch.bfloat16:
            return None
        if not current_platform.is_cuda() or not is_deep_gemm_supported():
            return None
        # Per-rank local weight shape [N_local, K_local].
        weight_shape = (output_size_per_partition, input_size_per_partition)
        if not should_use_deepgemm_for_fp8_linear(torch.bfloat16, weight_shape):
            return None

        kernel = init_fp8_linear_kernel(
            activation_quant_key=create_fp8_quant_key(
                static=False, group_shape=GroupShape(1, 128)
            ),
            weight_quant_key=create_fp8_quant_key(
                static=True, group_shape=GroupShape(128, 128)
            ),
            input_dtype=params_dtype,
            out_dtype=params_dtype,
            weight_shape=weight_shape,
            module_name=f"{self.layer_name} (GLM fp8-dense)",
        )
        logger.info_once(
            "GLM fp8-dense: routing INT8 dense linear %s (N=%d, K=%d) through %s",
            self.layer_name,
            output_size_per_partition,
            input_size_per_partition,
            type(kernel).__name__,
            scope="global",
        )
        return kernel

    def _convert_int8_to_fp8_block(self, layer: torch.nn.Module) -> None:
        # Dequantize the INT8 (uint8b128) weight-only checkpoint to bf16, then
        # requantize to 128x128-block-scaled FP8 (E4M3) so the DeepGEMM blockwise
        # kernel can consume it -- mirroring the zai GLM-5.2-FP8 weight format.
        packed = layer.weight_packed.data  # int32 [N, K // pack_factor]
        # weight_packed is stored output_dim=0, input_dim=1, packed along input.
        unpacked = unpack_quantized_values_into_int32(
            packed, self.quant_type, packed_dim=1
        )  # [N, K], unsigned values in [0, 255]
        q = unpacked.to(torch.int32) - self.quant_type.bias  # signed [-128, 127]

        scale = layer.weight_scale.data.to(torch.float32)  # [N, num_groups]
        n, k = q.shape
        num_groups = scale.shape[-1]
        assert k % num_groups == 0, (
            f"weight K={k} not divisible by scale groups={num_groups}"
        )
        group_size = k // num_groups
        w_bf16 = (
            q.to(torch.float32).view(n, num_groups, group_size)
            * scale.view(n, num_groups, 1)
        ).view(n, k).to(torch.bfloat16)

        use_e8m0 = is_deep_gemm_e8m0_used()
        w_fp8, w_scale = per_block_cast_to_fp8(
            w_bf16, block_size=[128, 128], use_ue8m0=use_e8m0
        )

        # Drop the INT8 checkpoint parameters and install the FP8-block ones.
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_shape",
            "weight_zero_point",
            "weight_g_idx",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        layer.weight_block_size = [128, 128]
        layer.register_parameter(
            "weight", torch.nn.Parameter(w_fp8, requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale_inv", torch.nn.Parameter(w_scale, requires_grad=False)
        )
        # Applies the DeepGEMM scale/weight layout (deepgemm_post_process).
        self.fp8_dense_kernel.process_weights_after_loading(layer)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.fp8_dense_kernel is not None:
            self._convert_int8_to_fp8_block(layer)
            return
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if self.fp8_dense_kernel is not None:
            return self.fp8_dense_kernel.apply_weights(layer, x, bias)
        return self.kernel.apply_weights(layer, x, bias)
