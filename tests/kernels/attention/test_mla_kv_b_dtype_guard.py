# SPDX-License-Identifier: Apache-2.0
"""Packed-quantized kv_b_proj weights must not drive a kv_c_normed upcast.

NVFP4 packs into uint8 and int-quant (compressed-tensors) packs into int32.
Both are containers, not real activation dtypes -- casting kv_c_normed to
them corrupts chunked prefill past ~4K prompts.
"""
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    is_packed_quantized_dtype,
)


def test_uint8_is_packed():
    assert is_packed_quantized_dtype(torch.uint8)


def test_int32_is_packed():
    assert is_packed_quantized_dtype(torch.int32)


def test_bfloat16_is_not_packed():
    assert not is_packed_quantized_dtype(torch.bfloat16)


def test_float16_is_not_packed():
    assert not is_packed_quantized_dtype(torch.float16)


def test_float8_is_not_packed():
    assert not is_packed_quantized_dtype(torch.float8_e4m3fn)
