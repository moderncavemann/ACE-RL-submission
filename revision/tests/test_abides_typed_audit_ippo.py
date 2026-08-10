"""Fast contract tests for typed-audit updates on the pinned official IPPO.

These tests deliberately avoid Docker and market rollouts.  They exercise the
observation-copy contract, the complete-batch set loss, and the disjoint
actor/critic update path on a two-step in-memory replay buffer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.agents.official_on_policy_mappo import (  # noqa: E402
    OFFICIAL_COMMIT,
    OfficialMAPPO,
    OfficialMAPPOConfig,
)
from src.agents.typed_audit import (  # noqa: E402
    TypedAuditSpec,
    build_audit_batch,
    official_update_with_set_audit,
    set_audit_loss_from_probs,
)


SOURCE = Path.home() / ".cache" / "ace-rl" / "on-policy" / OFFICIAL_COMMIT


def audit_spec(**overrides: object) -> TypedAuditSpec:
    values: dict[str, object] = {
        "acceptable_action_indices": (0, 1, 2),
        "shadow_action_index": 2,
        "public_volatility_max_ticks": 1.0,
        "absolute_inventory_max_units": 20.0,
        "actor_coefficient": 0.30,
        "action_cost_coefficient": 0.20,
        "probability_floor": 1e-12,
    }
    values.update(overrides)
    return TypedAuditSpec(**values)


def encoded_observations() -> np.ndarray:
    """Return one valid, one inventory-blocked, and one risk-blocked row."""

    obs = np.zeros((3, 7), dtype=np.float32)
    obs[:, 1] = 5.0 / 6.0  # previous action index 5 (six-tick half spread)
    obs[:, 2] = 6.0 / 7.0  # pre-audit population mean quote
    obs[:, 4] = 0.1  # finite flow-share feature, irrelevant to the audit rule
    obs[0, 0] = 0.0
    obs[0, 3] = 0.5 / 4.0  # 0.5 volatility ticks: valid
    obs[1, 0] = 50.0 / 100.0  # 50 inventory units: blocked
    obs[1, 3] = 0.5 / 4.0
    obs[2, 0] = 0.0
    obs[2, 3] = 2.0 / 4.0  # 2 volatility ticks: blocked
    return obs


def official_config() -> OfficialMAPPOConfig:
    return OfficialMAPPOConfig(
        source_root=str(SOURCE),
        seed=812,
        episode_steps=2,
        centralized_value=False,
        hidden_size=16,
        data_chunk_length=1,
        learning_rate=2e-2,
        critic_learning_rate=2e-2,
        entropy_coefficient=0.0,
        ppo_epoch=1,
        num_mini_batch=1,
        device="cpu",
    )


def ten_agent_observations() -> np.ndarray:
    rows = np.zeros((10, 7), dtype=np.float32)
    rows[:, 1] = 5.0 / 6.0
    rows[:, 2] = 6.0 / 7.0
    rows[:, 3] = 0.5 / 4.0
    rows[:, 4] = 0.1
    return rows


def filled_two_step_buffer(learner: OfficialMAPPO) -> object:
    obs = ten_agent_observations()
    critic_obs = obs.copy()
    buffer = learner.new_buffer(obs, critic_obs)
    rnn_actor, rnn_critic, masks = learner.initial_recurrent_states()
    uniforms = np.linspace(0.03, 0.97, 10, dtype=np.float32)
    for step in range(2):
        actions, logp, values, rnn_actor, rnn_critic = learner.act(
            obs,
            critic_obs,
            rnn_actor,
            rnn_critic,
            masks,
            uniforms,
        )
        next_obs = obs.copy()
        # Keep five valid rows and make five inventory-blocked rows.  This
        # checks the full-batch denominator and invalid-row zeroing in update.
        next_obs[5:, 0] = 50.0 / 100.0
        terminal_masks = (
            np.zeros_like(masks) if step == 1 else np.ones_like(masks)
        )
        learner.insert(
            buffer,
            next_obs,
            next_obs,
            rnn_actor,
            rnn_critic,
            actions,
            logp,
            values,
            np.linspace(-0.2, 0.2, 10, dtype=np.float32),
            terminal_masks,
        )
        obs = next_obs
        critic_obs = next_obs
        masks = terminal_masks
    return buffer


def test_shadow_copy_changes_only_encoded_action_and_population_mean() -> None:
    spec = audit_spec()
    obs = encoded_observations()
    original = obs.copy()

    batch = build_audit_batch(obs, spec, always_on=False)

    np.testing.assert_array_equal(obs, original)
    assert set(batch) == {
        "shadow_obs",
        "valid_mask",
        "response_mask",
        "measurement_response_mask",
    }
    shadow = batch["shadow_obs"]
    assert shadow is not obs
    np.testing.assert_array_equal(batch["valid_mask"], [True, False, False])
    expected_response = np.array([True, True, True, False, False, False, False])
    expected_measurement = np.broadcast_to(expected_response, (3, 7))
    np.testing.assert_array_equal(
        batch["measurement_response_mask"], expected_measurement
    )
    np.testing.assert_array_equal(batch["response_mask"][0], expected_response)
    np.testing.assert_array_equal(
        batch["response_mask"][1:], np.zeros((2, 7), dtype=bool)
    )

    untouched = [0, 3, 4, 5, 6]
    np.testing.assert_array_equal(shadow[:, untouched], original[:, untouched])
    np.testing.assert_allclose(shadow[:, 1], spec.shadow_action_index / 6.0)
    expected_mean = original[:, 2] + (
        spec.shadow_action_index - original[:, 1] * 6.0
    ) / (spec.n_agents * spec.mean_quote_scale_ticks)
    np.testing.assert_allclose(shadow[:, 2], expected_mean, rtol=0.0, atol=1e-7)


def test_full_invalid_mask_is_exact_zero_with_zero_gradient() -> None:
    probabilities = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.10, 0.10, 0.10, 0.10],
            [0.20, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = np.array([False, False])
    response = np.broadcast_to(
        np.array([True, True, True, False, False, False, False]), (2, 7)
    ).copy()

    loss = set_audit_loss_from_probs(probabilities, valid, response)
    assert loss.item() == 0.0
    loss.backward()
    assert probabilities.grad is not None
    torch.testing.assert_close(
        probabilities.grad, torch.zeros_like(probabilities.grad), rtol=0.0, atol=0.0
    )


def test_complete_batch_formula_and_probability_floor_are_finite() -> None:
    probabilities = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.10, 0.10, 0.10, 0.10],
            [0.20, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15],
        ],
        dtype=torch.float64,
    )
    valid = np.array([True, False])
    response = np.broadcast_to(
        np.array([True, True, True, False, False, False, False]), (2, 7)
    ).copy()
    loss = set_audit_loss_from_probs(probabilities, valid, response)
    assert loss.item() == pytest.approx(-np.log(0.60) / 2.0)

    tiny = torch.tensor(
        [[1e-30, 1e-30, 1e-30, 0.25, 0.25, 0.25, 0.25]],
        dtype=torch.float64,
    )
    tiny_loss = set_audit_loss_from_probs(
        tiny,
        np.array([True]),
        response[:1],
        probability_floor=1e-12,
    )
    assert torch.isfinite(tiny_loss)
    assert tiny_loss.item() == pytest.approx(-np.log(1e-12))


def test_always_on_marks_every_row_valid() -> None:
    batch = build_audit_batch(encoded_observations(), audit_spec(), always_on=True)
    np.testing.assert_array_equal(batch["valid_mask"], np.ones(3, dtype=bool))
    assert np.all(batch["response_mask"].sum(axis=-1) == 3)


def test_audit_only_actor_step_increases_response_mass_and_isolates_critic() -> None:
    learner = OfficialMAPPO(official_config())
    spec = audit_spec()
    obs = ten_agent_observations()
    audit = build_audit_batch(obs, spec)
    rnn_actor, rnn_critic, masks = learner.initial_recurrent_states()
    critic_before = {
        name: value.detach().clone()
        for name, value in learner.policy.critic.state_dict().items()
    }
    with torch.no_grad():
        value_before, _ = learner.policy.critic(obs, rnn_critic, masks)
        value_before = value_before.detach().clone()
    before = learner.action_probabilities(audit["shadow_obs"], rnn_actor, masks)
    before_mass = float(before[:, list(spec.acceptable_action_indices)].sum(axis=-1).mean())

    learner.policy.actor_optimizer.zero_grad(set_to_none=True)
    learner.policy.critic_optimizer.zero_grad(set_to_none=True)
    actor = learner.policy.actor
    shadow_tensor = torch.as_tensor(audit["shadow_obs"], dtype=torch.float32)
    features = actor.base(shadow_tensor)
    features, _ = actor.rnn(
        features,
        torch.as_tensor(rnn_actor, dtype=torch.float32),
        torch.as_tensor(masks, dtype=torch.float32),
    )
    probabilities = actor.act.get_probs(features)
    loss = set_audit_loss_from_probs(
        probabilities,
        audit["valid_mask"],
        audit["response_mask"],
        probability_floor=spec.probability_floor,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in learner.policy.actor.parameters()
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in learner.policy.critic.parameters()
    )
    learner.policy.actor_optimizer.step()

    after = learner.action_probabilities(audit["shadow_obs"], rnn_actor, masks)
    after_mass = float(after[:, list(spec.acceptable_action_indices)].sum(axis=-1).mean())
    assert after_mass > before_mass
    np.testing.assert_allclose(after.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6)
    for name, value in learner.policy.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[name], rtol=0.0, atol=0.0)
    with torch.no_grad():
        value_after, _ = learner.policy.critic(obs, rnn_critic, masks)
    torch.testing.assert_close(value_after, value_before, rtol=0.0, atol=0.0)


def test_official_ippo_joint_update_reports_exact_audit_invariants() -> None:
    learner = OfficialMAPPO(official_config())
    buffer = filled_two_step_buffer(learner)
    metrics = official_update_with_set_audit(
        learner,
        buffer,
        update_seed=919,
        spec=audit_spec(),
        always_on=False,
    )

    required = {
        "value_loss",
        "policy_loss",
        "dist_entropy",
        "actor_grad_norm",
        "critic_grad_norm",
        "ratio",
        "audit_loss",
        "audit_weighted_loss",
        "audit_response_mass",
        "audit_actor_coefficient",
        "audit_total_rows",
        "audit_valid_rows",
        "audit_valid_rate",
        "audit_valid_rows_seen",
        "audit_zero_valid_minibatches",
        "audit_invalid_loss_max_abs",
        "audit_critic_grad_max_abs",
    }
    assert required.issubset(metrics)
    assert all(np.isfinite(float(metrics[key])) for key in required)
    assert metrics["audit_total_rows"] == 20.0
    assert metrics["audit_valid_rows"] == 15.0
    assert metrics["audit_valid_rate"] == pytest.approx(0.75)
    assert metrics["audit_loss"] > 0.0
    assert 0.0 < metrics["audit_response_mass"] < 1.0
    assert metrics["audit_invalid_loss_max_abs"] == 0.0
    assert metrics["audit_critic_grad_max_abs"] == 0.0
