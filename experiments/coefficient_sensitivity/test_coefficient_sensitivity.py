#!/usr/bin/env python3
"""Unit checks for the frozen audit-coefficient sensitivity runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("alpha_sensitivity", HERE / "run_experiment.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_hash_is_frozen() -> None:
    assert MODULE.sha256(HERE / "PROTOCOL.md") == MODULE.EXPECTED_PROTOCOL_SHA256


def test_grid_contains_reference_and_declared_value() -> None:
    assert MODULE.ALPHAS[0] == 0.0
    assert 0.30 in MODULE.ALPHAS
    assert tuple(sorted(MODULE.ALPHAS)) == MODULE.ALPHAS


def test_formal_seed_count_and_disjoint_range() -> None:
    assert len(MODULE.FORMAL_SEEDS) == 50
    assert MODULE.FORMAL_SEEDS[0] == 6000
    assert MODULE.FORMAL_SEEDS[-1] == 6049
