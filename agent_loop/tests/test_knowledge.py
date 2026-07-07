"""
Tests for the qlib capability substrate (ADR 0002 Phase F).

Reads the generated/curated files only (no qlib needed). Verifies the substrate is present and that the
real base features (ALPHA20) are fully grounded in it — i.e. every operator and $field the loop actually
uses is documented, so a factor built from documented pieces references real qlib capabilities.
"""
from __future__ import annotations

import re
from pathlib import Path

from rdagent.utils.qlib import ALPHA20

KDIR = Path(__file__).resolve().parents[1] / "knowledge"
FILES = ["operators.md", "fields.md", "handlers.md", "metrics.md", "models.md", "README.md"]


def _documented_ops() -> set[str]:
    return set(re.findall(r"\| `([A-Z][A-Za-z]+)\(", (KDIR / "operators.md").read_text()))


def _documented_fields() -> set[str]:
    # only the present-fields table, not the "Absent fields" caveat section
    main = (KDIR / "fields.md").read_text().split("## ⚠️")[0]
    return set(re.findall(r"\| `\$([a-z]+)`", main))


def test_substrate_files_present_and_nontrivial():
    for f in FILES:
        p = KDIR / f
        assert p.exists() and len(p.read_text()) > 200, f


def test_alpha20_base_is_grounded_in_substrate():
    ops, fields = _documented_ops(), _documented_fields()
    used_ops, used_fields = set(), set()
    for expr in ALPHA20.values():
        used_ops |= set(re.findall(r"([A-Z][A-Za-z]+)\(", expr))
        used_fields |= set(re.findall(r"\$([a-z]+)", expr))
    assert used_ops <= ops, f"ALPHA20 uses undocumented operators: {used_ops - ops}"
    assert used_fields <= fields, f"ALPHA20 uses undocumented fields: {used_fields - fields}"


def test_absent_field_caveat_present():
    text = (KDIR / "fields.md").read_text()
    assert "vwap" in text.lower() and "DO NOT USE" in text
    assert "vwap" not in _documented_fields()  # not listed as an available field
