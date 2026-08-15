# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requantise a draft model's lm_head to FP8. Speed only -- it cannot change output.

A speculative draft is a PROPOSAL. Every drafted token is either accepted by the target's
own rejection test or replaced by a token sampled from the target's distribution, so the
served distribution is a property of the target alone. Degrading the drafter can only cost
acceptance (speed); it can never change what the model says. That makes the drafter the one
place in the stack where precision may be traded freely.

And on an MTP drafter the lm_head is the dominant cost, which is easy to miss because the
drafter is "just one layer":

    lm_head            154,880 x 6,144 bf16 = 1.9 GB, 476 MB per rank at TP=4
    read               once per drafter pass, so num_speculative_tokens times per step
    at k=3             3 x 476 MB = 1.4 GB/rank/step, ~7% of ALL bytes a decode step streams

The MTP block does not alias the target's head -- the loader copies `lm_head.weight` into
`model.layers.{spec}.shared_head.head.weight` (glm4_moe_mtp.py), so it is a second full
copy. Requantising it therefore *frees* memory rather than costing any, which matters on
unified-memory parts that idle with a couple of GiB spare.

Hooked on the draft model only. The target's head is never touched.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


class Fp8DraftHeadMethod:
    """Drop-in replacement for the head's ``quant_method``.

    Only ``apply`` is exercised at runtime; weight creation and loading already happened
    in bf16 before this swap, which is deliberate -- it keeps the checkpoint path, the
    vocab-parallel sharding and the tensor-parallel gather completely untouched. Only the
    matmul changes.
    """

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor):
        from vllm import _custom_ops as ops

        self.weight_fp8 = weight_fp8            # [num_embeddings_per_rank, hidden]
        # Stored pre-transposed to [1, N]: cutlass_scaled_mm broadcasts scale_b against a
        # [K, N] operand, and reshaping per call would be pure overhead on the decode path.
        self.weight_scale = weight_scale.reshape(1, -1).contiguous()
        # Bound once rather than imported per call -- this runs on every drafter step.
        self._scaled_mm = ops.cutlass_scaled_mm

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"FP8 draft head needs a bf16/fp16 hidden state, got {x.dtype}; "
                "cutlass_scaled_mm cannot emit fp32."
            )
        flat = x.reshape(-1, x.shape[-1])
        # Per-token activation scale. Decode drafts one token per sequence, so this is a
        # handful of rows -- the quantise is far cheaper than the weight read it enables.
        x_amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        x_scale = (x_amax / FP8_MAX).to(torch.float32)
        x_fp8 = (flat / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        # vLLM's Cutlass FP8 rather than torch._scaled_mm: this runs at M=1 on the decode
        # path, where _scaled_mm's alignment requirements are version-dependent, and the
        # Cutlass op is the same family already serving this deployment's dense layers on
        # SM121. Weight is stored [vocab_rank, hidden]; the op wants [hidden, vocab_rank].
        out = self._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])


ROWS_PER_CHUNK = 4096


def _quantize_rowwise(wd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-scaled FP8 copy of ``wd``, built in chunks.

    Chunked because the obvious one-liner is not affordable where this matters most. A
    154,880 x 6,144 bf16 head is 454 MB per rank; promoting it whole to fp32 to divide by
    the scale would allocate ~908 MB, and the clamp another, i.e. a transient ~2 GB spike on
    a unified-memory part that can be idling with ~1 GiB free. Chunking caps the spike at
    ROWS_PER_CHUNK x hidden x 4 bytes (~100 MB) regardless of vocabulary size.

    Per-OUTPUT-ROW scaling, not per-tensor: a handful of high-magnitude vocabulary rows would
    otherwise set one global scale and crush the resolution of every other token's logit.
    """
    out = torch.empty_like(wd, dtype=FP8_DTYPE)
    scale = torch.empty(wd.shape[0], 1, dtype=torch.float32, device=wd.device)
    for lo in range(0, wd.shape[0], ROWS_PER_CHUNK):
        hi = min(lo + ROWS_PER_CHUNK, wd.shape[0])
        block = wd[lo:hi].to(torch.float32)
        s = (block.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
        out[lo:hi] = (block / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        scale[lo:hi] = s
        del block, s
    return out, scale


def quantize_draft_lm_head_fp8(model: torch.nn.Module) -> int:
    """Swap every ParallelLMHead in ``model`` to FP8. Returns bytes freed."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    # Tied word embeddings share ONE storage between lm_head and embed_tokens. Quantising is
    # still safe there (we build a separate FP8 tensor and only swap the matmul), but FREEING
    # the bf16 copy would delete the input embedding and destroy the model. Detect sharing by
    # storage rather than by reading a config flag, so a model that ties without advertising
    # it is still handled.
    shared: dict[int, int] = {}
    for p in model.parameters():
        if p is not None and p.device.type != "meta":
            shared[p.data_ptr()] = shared.get(p.data_ptr(), 0) + 1

    freed = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, ParallelLMHead):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dtype not in (torch.bfloat16, torch.float16):
            continue

        wd = w.data
        w_fp8, scale = _quantize_rowwise(wd)
        before = wd.numel() * wd.element_size()
        after = w_fp8.numel() * w_fp8.element_size() + scale.numel() * scale.element_size()
        mod.quant_method = Fp8DraftHeadMethod(w_fp8, scale)

        tied = shared.get(wd.data_ptr(), 1) > 1
        if tied:
            # Keep the bf16 copy: something else (the input embedding) still needs it. The
            # traffic saving still applies -- that comes from the matmul, not from freeing.
            logger.info(
                "Draft lm_head %s is TIED to another parameter: using FP8 for the matmul "
                "but keeping the bf16 copy (%.0f MB not freed)", name, before / 2**20
            )
        else:
            # Drop the bf16 copy, so this FREES memory rather than costing it. Anything still
            # reading `.weight` afterwards will fail loudly, which is what we want -- a silent
            # fallback to the bf16 path would report a speedup that never happened.
            mod.register_parameter("weight", None)
            freed += before - after
        del wd, w
        logger.info(
            "Draft lm_head %s requantised to FP8: %.0f MB -> %.0f MB",
            name, before / 2**20, after / 2**20,
        )

    if freed:
        torch.cuda.empty_cache()
    return freed
