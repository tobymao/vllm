# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step speculative depth chosen from draft confidence AND verify cost.

On a dense model an extra verify row is nearly free: every row reuses the same weight
matrices, so deeper speculation costs little and a depth policy can ignore cost entirely.
That is the assumption behind ``AcceptanceLengthController``, whose rule is
``floor(mean_accepted + 1.5)`` -- no cost term anywhere.

On a sparse MoE it is false. The verification footprint is the size of the UNION of experts
the verified tokens route to, so each additional row drags in roughly ``num_experts_per_tok``
more experts and their weights must be streamed. Depth is bought with memory bandwidth, and
whether it is worth buying depends on how confident the drafter is *on this step*.

This picks the depth that maximises expected tokens per millisecond, per step:

    E[tokens](k) = 1 + sum_{i<=k} s_i          s_i = prod_{j<=i} q_j, the drafter's own
                                               probability that its draft survives to i
    cost(k)      = fixed + moe * U(k+1)/U(R) + per_pass * K

    U(r) = 1 - (1 - E_act/E_tot)^r             expected fraction of experts touched by r
                                               independently-routing rows

``U`` is the standard occupancy curve: each row draws ``E_act`` of ``E_tot`` experts, so the
union saturates rather than growing linearly, and rows get cheaper as depth grows. That
curvature is exactly what a cost-blind rule cannot see.

Losslessness: this only ever REMOVES draft proposals before verification. The verification
rule, the acceptance test and the target distribution are untouched, so output is unaffected
in distribution -- it is a speed knob and nothing else.

Why depth is chosen once per step for the whole batch rather than per request: vLLM captures
one uniform-decode cudagraph per depth (see ``gpu/cudagraph_utils.py``). Ragged per-request
depths would miss every captured shape and fall back to the slow path, losing more than the
policy gains. At batch 1 -- the single-stream agentic case this targets -- the two coincide.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-step cost of verifying ``rows`` tokens, in milliseconds.

    Calibrate on the target cluster: run two fixed depths, read step time as
    ``median_TPOT * acceptance_length`` from a real benchmark (NOT from a profiler trace --
    CUPTI inflates the step), and solve for ``fixed_ms`` and ``moe_ms``.
    """

    fixed_ms: float
    """Everything independent of row count: dense GEMMs, collectives, attention."""
    moe_ms: float
    """MoE expert streaming at ``ref_rows`` rows."""
    ref_rows: int
    """Row count at which ``moe_ms`` was measured."""
    per_pass_ms: float
    """One drafter forward. Paid ``max_depth`` times regardless of the depth chosen: the
    drafting loop is cudagraph-captured, so it cannot exit early on a data-dependent test."""
    experts_active: int
    """``num_experts_per_tok``."""
    experts_total: int
    """``n_routed_experts``."""

    def union(self, rows: int) -> float:
        miss = 1.0 - self.experts_active / self.experts_total
        return 1.0 - miss**rows

    def ms(self, depth: int, max_depth: int) -> float:
        scale = self.union(depth + 1) / self.union(self.ref_rows)
        return self.fixed_ms + self.moe_ms * scale + self.per_pass_ms * max_depth


def choose_depth(
    survival: list[float],
    cost: CostModel,
    max_depth: int,
    min_depth: int = 1,
) -> int:
    """Depth in ``[min_depth, max_depth]`` maximising expected tokens per millisecond.

    ``survival[i]`` is the drafter's estimated probability that draft tokens 0..i are all
    accepted -- a running product of per-position draft probabilities, so it is
    non-increasing. Only the leading ``max_depth`` entries are considered.
    """
    if max_depth <= min_depth:
        return max(min_depth, 1)

    best_depth, best_rate = min_depth, -1.0
    expected = 1.0
    for depth in range(1, max_depth + 1):
        expected += survival[depth - 1] if depth - 1 < len(survival) else 0.0
        if depth < min_depth:
            continue
        rate = expected / cost.ms(depth, max_depth)
        if rate > best_rate:
            best_depth, best_rate = depth, rate
    return best_depth


def running_survival(probs: list[float]) -> list[float]:
    """Per-position draft probabilities -> cumulative survival."""
    out, acc = [], 1.0
    for p in probs:
        acc *= p
        out.append(acc)
    return out
