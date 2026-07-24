# SPDX-License-Identifier: Apache-2.0
"""The MTP draft quant config must inherit the target's fused-module mappings."""
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.deepseek_mtp import (
    seed_draft_packed_modules_mapping,
)


def _quant_config(mapping=None):
    return SimpleNamespace(packed_modules_mapping=dict(mapping or {}))


def test_seeds_gate_up_proj():
    qc = _quant_config()
    seed_draft_packed_modules_mapping(qc, SimpleNamespace(q_lora_rank=None))
    assert qc.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]


def test_seeds_fused_qkv_when_q_lora_rank_set():
    qc = _quant_config()
    seed_draft_packed_modules_mapping(qc, SimpleNamespace(q_lora_rank=1536))
    assert qc.packed_modules_mapping["fused_qkv_a_proj"] == [
        "q_a_proj",
        "kv_a_proj_with_mqa",
    ]


def test_omits_fused_qkv_without_q_lora_rank():
    qc = _quant_config()
    seed_draft_packed_modules_mapping(qc, SimpleNamespace(q_lora_rank=None))
    assert "fused_qkv_a_proj" not in qc.packed_modules_mapping


def test_never_clobbers_existing_mapping():
    qc = _quant_config({"gate_up_proj": ["preexisting"]})
    seed_draft_packed_modules_mapping(qc, SimpleNamespace(q_lora_rank=1536))
    assert qc.packed_modules_mapping["gate_up_proj"] == ["preexisting"]


@pytest.mark.parametrize("qc", [None, SimpleNamespace()])
def test_tolerates_missing_config_or_attribute(qc):
    seed_draft_packed_modules_mapping(qc, SimpleNamespace(q_lora_rank=1536))
