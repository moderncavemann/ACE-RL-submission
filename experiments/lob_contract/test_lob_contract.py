from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ORIGINAL_PROJECT = HERE.parents[1] / "ace_rl_project"
os.environ.setdefault("ACE_RL_PROJECT_ROOT", str(ORIGINAL_PROJECT))
SPEC = importlib.util.spec_from_file_location("original_lob_contract", HERE / "run_experiment.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_original_engine_hash() -> None:
    assert MODULE.sha256(MODULE.ENGINE_PATH) == MODULE.EXPECTED_ENGINE_SHA256


def test_valid_contract_direction_and_live_isolation() -> None:
    cfg = MODULE.Config()
    spec = MODULE.SCENARIOS[0]
    state = MODULE.make_state(5000, spec, cfg)
    before = MODULE.live_state_hash(state)
    audit = MODULE.audit_state(state, 5000, spec.name, cfg)
    update = MODULE.audit_actor_step(np.zeros(7), audit, cfg)
    assert audit.z == 1
    assert audit.response_ticks == (1, 2, 3)
    assert all(row["action_feasible"] for row in audit.action_checks)
    assert update["response_probability_delta"] > 0.0
    assert MODULE.live_state_hash(state) == before


def test_invalid_contract_is_exact_zero() -> None:
    cfg = MODULE.Config()
    for spec in MODULE.SCENARIOS[1:]:
        state = MODULE.make_state(5001, spec, cfg)
        audit = MODULE.audit_state(state, 5001, spec.name, cfg)
        update = MODULE.audit_actor_step(np.linspace(-0.2, 0.2, 7), audit, cfg)
        assert audit.z == 0
        assert update["audit_loss"] == 0.0
        assert update["max_abs_logit_change"] == 0.0
        assert update["max_abs_probability_change"] == 0.0


def test_smoke_run_passes(tmp_path: Path) -> None:
    result = MODULE.run_once(tmp_path, MODULE.SMOKE_SEEDS, "engineering_smoke")
    assert result["summary"]["scenario_row_count"] == 6
    assert result["summary"]["failure_count"] == 0
    assert result["summary"]["formal_status"] == "PASS"
