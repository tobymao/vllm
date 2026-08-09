# SPDX-License-Identifier: Apache-2.0
"""Selection tests for the B12X (sparkinfer) WNA16 MoE backend.

Covers the parts of the oracle that are independent of the sparkinfer weight
prep: the enum, the user-facing `--moe-backend b12x` string map, and the
auto-selection priority (B12X must rank ahead of MARLIN on SM12x when
sparkinfer is importable, and must be absent everywhere else).

CPU-only: the platform is monkeypatched, no GPU or sparkinfer install needed.
"""
import pytest

from vllm.model_executor.layers.fused_moe.oracle import int_wna16
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    map_wna16_backend,
)


class TestB12xWNA16Oracle:
    def test_enum_has_b12x(self):
        assert WNA16MoEBackend.B12X.value == "B12X"

    def test_moe_backend_string_maps_to_b12x(self):
        """`--moe-backend b12x` must resolve, not raise."""
        assert map_wna16_backend("b12x") == WNA16MoEBackend.B12X

    def test_unknown_backend_still_raises_and_lists_b12x(self):
        with pytest.raises(ValueError) as exc:
            map_wna16_backend("definitely-not-a-backend")
        assert "b12x" in str(exc.value)

    def test_priority_puts_b12x_ahead_of_marlin_on_sm12x(self, monkeypatch):
        monkeypatch.setattr(int_wna16, "_b12x_wna16_available", lambda: True)
        backends = int_wna16._get_priority_backends()
        assert WNA16MoEBackend.B12X in backends
        assert backends.index(WNA16MoEBackend.B12X) < backends.index(
            WNA16MoEBackend.MARLIN
        )

    def test_b12x_absent_when_unavailable(self, monkeypatch):
        """Wrong arch, or sparkinfer not importable -> never offered, never crash."""
        monkeypatch.setattr(int_wna16, "_b12x_wna16_available", lambda: False)
        backends = int_wna16._get_priority_backends()
        assert WNA16MoEBackend.B12X not in backends
        assert WNA16MoEBackend.MARLIN in backends

    def test_availability_probe_is_crash_safe(self, monkeypatch):
        """A sparkinfer import explosion must degrade to False, not propagate."""
        def boom(*a, **k):
            raise RuntimeError("sparkinfer exploded")

        monkeypatch.setattr(int_wna16, "_sparkinfer_wna16_importable", boom)
        assert int_wna16._b12x_wna16_available() is False
