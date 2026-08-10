"""Fast, Docker-free contract tests for the typed-audit experiment runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from run_abides_typed_audit_ippo_v1 import (  # noqa: E402
    METHODS,
    PRIMARY_METRICS,
    aggregate_seed_rows,
    rollout_arm,
    runtime_config,
    validate_namespace_isolation,
)
from src.agents.typed_audit import TypedAuditSpec  # noqa: E402


def frozen_config() -> dict[str, Any]:
    return json.loads(
        (
            REVISION_ROOT / "configs" / "abides_typed_audit_ippo_v2.json"
        ).read_text(encoding="utf-8")
    )


def test_smoke_and_formal_runtime_namespaces_are_exact_and_disjoint() -> None:
    config = frozen_config()
    smoke = runtime_config(config, "engineering_smoke")
    formal = runtime_config(config, "formal")

    assert smoke == {
        "mode": "engineering_smoke",
        "policy_seeds": [560000],
        "response_episodes": 2,
        "evaluation_seeds": [711000000],
        "response_environment_seed_namespace": 710000000,
        "training_action_namespace": 710100,
        "evaluation_action_namespace": 710200,
        "workers": 1,
        "output_subdirectory": "engineering_smoke",
    }
    assert formal["policy_seeds"] == list(range(560000, 560010))
    assert formal["response_episodes"] == 96
    assert formal["evaluation_seeds"] == list(range(701000000, 701000020))
    assert formal["response_environment_seed_namespace"] == 700000000
    assert formal["training_action_namespace"] == 702000
    assert formal["evaluation_action_namespace"] == 703000
    assert formal["output_subdirectory"] == "formal"
    assert smoke["output_subdirectory"] != formal["output_subdirectory"]
    assert set(smoke["evaluation_seeds"]).isdisjoint(formal["evaluation_seeds"])
    assert {
        smoke["training_action_namespace"],
        smoke["evaluation_action_namespace"],
    }.isdisjoint(
        {
            formal["training_action_namespace"],
            formal["evaluation_action_namespace"],
        }
    )


def test_namespace_helper_reports_exact_frozen_path_counts() -> None:
    receipt = validate_namespace_isolation(frozen_config())
    assert receipt == {
        "formal_training_path_count": 10 * 96,
        "formal_evaluation_path_count": 20,
        "smoke_training_path_count": 2,
        "smoke_evaluation_path_count": 1,
        "environment_seed_sets_pairwise_disjoint": True,
        "formal_smoke_action_namespaces_disjoint": True,
        "within_mode_action_namespaces_disjoint": True,
    }
    assert 10 * 96 * len(METHODS) == 3840
    assert 10 * 20 * len(METHODS) == 800
    assert 1 * 2 * len(METHODS) == 8
    assert 1 * 1 * len(METHODS) == 4


class FakeLearner:
    def __init__(self) -> None:
        self.config = SimpleNamespace(n_agents=10, n_actions=7)

    def initial_recurrent_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.zeros((10, 1, 1), dtype=np.float32)
        masks = np.ones((10, 1), dtype=np.float32)
        return hidden.copy(), hidden.copy(), masks

    def action_probabilities(
        self, obs: np.ndarray, rnn_actor: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        del obs, rnn_actor, masks
        # The fixed acceptable set {0,1,2} has mass 0.30 on every row.  An
        # invalid-row typed response mask is empty, so q_X=0.30 proves that the
        # runner measures with measurement_response_mask instead.
        row = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4], dtype=np.float32)
        return np.repeat(row[None, :], 10, axis=0)

    def act(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        rnn_actor: np.ndarray,
        rnn_critic: np.ndarray,
        masks: np.ndarray,
        uniforms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del obs, critic_obs, masks, uniforms
        actions = np.full((10, 1), 3, dtype=np.int64)
        zeros = np.zeros((10, 1), dtype=np.float32)
        return actions, zeros.copy(), zeros.copy(), rnn_actor.copy(), rnn_critic.copy()

    def model_digest(self) -> str:
        return "fake-model"


class FakeEnvironment:
    def __init__(self) -> None:
        observations = np.zeros((10, 6), dtype=np.float32)
        observations[5:, 0] = 0.5  # 50 inventory units: canonical invalid rows.
        self.observations = observations
        self.step_kwargs: dict[str, Any] | None = None
        self.closed = False

    def reset(self, seed: int) -> dict[str, Any]:
        del seed
        return {
            "observations": self.observations.tolist(),
            "info": {},
            "done": False,
        }

    def step(self, actions: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del actions
        self.step_kwargs = dict(kwargs)
        return {
            "observations": self.observations.tolist(),
            # Large magnitude exposes float32 round-off in the redundant
            # raw-unit form while the scaled learner formula remains exact.
            "rewards": [-100000.0] * 10,
            "done": True,
            "info": {
                "path_digest": "fixed-path",
                "routing_enabled": False,
                "routing_total_variation": 0.0,
                "realized_targeted_quantity": 0.0,
                "inventory_penalty_by_dealer": [2.0] * 10,
                "inventory_by_dealer": [0.0] * 10,
                "mean_execution_half_spread": 4.0,
                "fill_ratio": 1.0,
                "realized_flow_quantity": 20.0,
                "customer_execution_cost_tick_units": 80.0,
            },
        }

    def close_episode(self) -> None:
        self.closed = True


def test_rollout_measures_qx_with_fixed_mask_and_applies_scaled_action_cost() -> None:
    config = frozen_config()
    config["episode_steps"] = 1
    environment = FakeEnvironment()
    result = rollout_arm(
        env=environment,
        learner=FakeLearner(),
        config=config,
        spec=TypedAuditSpec(**config["audit"]),
        fingerprint="test-fingerprint",
        runtime={"mode": "engineering_smoke"},
        phase="test",
        policy_seed=560000,
        environment_seed=770000000,
        episode=0,
        method="action_cost",
        action_namespace=710100,
        train=False,
        update_seed=0,
        base_model_digest="fake-base",
    )

    summary = result["summary"]
    assert summary["q_v"] == pytest.approx(0.30)
    assert summary["q_x"] == pytest.approx(0.30)
    assert summary["selectivity"] == pytest.approx(0.0)
    assert summary["valid_audit_rows"] == 5
    assert summary["invalid_audit_rows"] == 5
    assert summary["total_audit_rows"] == 10
    record = result["records"][0]
    np.testing.assert_allclose(record["financial_reward_raw"], [-100000.0] * 10)
    np.testing.assert_allclose(record["inventory_penalty"], [2.0] * 10)
    np.testing.assert_allclose(
        record["gross_pnl_before_inventory_penalty"], [-99998.0] * 10
    )
    np.testing.assert_allclose(record["action_cost"], [1.0] * 5 + [0.0] * 5)
    np.testing.assert_allclose(
        record["learning_reward_scaled"], [-1000.2] * 5 + [-1000.0] * 5,
        atol=1e-4,
    )
    assert environment.step_kwargs == {
        "gate_active": False,
        "routing_enabled": False,
        "routing_strength": 0.0,
    }
    assert environment.closed is True


def per_path_row(path_seed: int, method: str, q_v: float, valid: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        metric: 1.0 for metric in PRIMARY_METRICS
    }
    row.update(
        {
            "method": method,
            "evaluation_seed": path_seed,
            "q_v": q_v,
            "q_x": 0.2,
            "selectivity": q_v - 0.2,
            "valid_audit_rate": valid / 10.0,
            "valid_audit_rows": valid,
            "invalid_audit_rows": 10 - valid,
            "total_audit_rows": 10,
        }
    )
    return row


def test_seed_summary_uses_retained_audit_denominators() -> None:
    paths = []
    for method in METHODS:
        paths.append(per_path_row(1, method, q_v=0.1, valid=1))
        paths.append(per_path_row(2, method, q_v=0.9, valid=9))
    rows = aggregate_seed_rows(
        [
            {
                "policy_seed": 560000,
                "evaluation_seeds": [1, 2],
                "evaluation": {"per_path": paths},
            }
        ],
        METHODS,
    )
    assert len(rows) == 1
    for method in METHODS:
        assert rows[0][f"{method}__q_v"] == pytest.approx(0.82)
        assert rows[0][f"{method}__q_x"] == pytest.approx(0.2)
        assert rows[0][f"{method}__selectivity"] == pytest.approx(0.62)
        assert rows[0][f"{method}__valid_audit_rows"] == 10
        assert rows[0][f"{method}__invalid_audit_rows"] == 10
        assert rows[0][f"{method}__total_audit_rows"] == 20


def test_seed_summary_fails_closed_on_zero_audit_denominator() -> None:
    paths = []
    for method in METHODS:
        row = per_path_row(1, method, q_v=0.1, valid=1)
        row.update(
            {
                "q_v": None,
                "q_x": None,
                "selectivity": None,
                "valid_audit_rows": 0,
                "invalid_audit_rows": 0,
                "total_audit_rows": 0,
            }
        )
        paths.append(row)
    with pytest.raises(RuntimeError, match="complete audit denominator"):
        aggregate_seed_rows(
            [
                {
                    "policy_seed": 560000,
                    "evaluation_seeds": [1],
                    "evaluation": {"per_path": paths},
                }
            ],
            METHODS,
        )
