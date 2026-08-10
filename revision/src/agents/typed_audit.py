"""Typed-audit actor updates for the pinned official recurrent IPPO learner.

The audit term is evaluated on discarded shadow observations.  It is added only
to the actor objective; PPO returns and the value loss remain the financial
quantities stored in the official replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TypedAuditSpec:
    """Frozen ABIDES typed-audit rule and observation encoding."""

    acceptable_action_indices: tuple[int, ...] = (0, 1, 2)
    shadow_action_index: int = 2
    public_volatility_max_ticks: float = 1.0
    absolute_inventory_max_units: float = 20.0
    actor_coefficient: float = 0.30
    action_cost_coefficient: float = 0.20
    probability_floor: float = 1e-12
    n_actions: int = 7
    n_agents: int = 10
    inventory_feature_index: int = 0
    previous_action_feature_index: int = 1
    mean_quote_feature_index: int = 2
    public_volatility_feature_index: int = 3
    inventory_scale_units: float = 100.0
    public_volatility_scale_ticks: float = 4.0
    mean_quote_scale_ticks: float = 7.0

    def __post_init__(self) -> None:
        acceptable = tuple(int(index) for index in self.acceptable_action_indices)
        object.__setattr__(self, "acceptable_action_indices", acceptable)
        if self.n_actions <= 1:
            raise ValueError("typed audit requires at least two actions")
        if self.n_agents <= 0:
            raise ValueError("typed audit requires a positive agent count")
        if not acceptable or len(set(acceptable)) != len(acceptable):
            raise ValueError("acceptable action indices must be non-empty and unique")
        if any(index < 0 or index >= self.n_actions for index in acceptable):
            raise ValueError("acceptable action index is outside the action space")
        if self.shadow_action_index not in acceptable:
            raise ValueError("shadow action must belong to the acceptable response set")
        numeric_nonnegative = {
            "public_volatility_max_ticks": self.public_volatility_max_ticks,
            "absolute_inventory_max_units": self.absolute_inventory_max_units,
            "actor_coefficient": self.actor_coefficient,
            "action_cost_coefficient": self.action_cost_coefficient,
        }
        for name, value in numeric_nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        positive = {
            "probability_floor": self.probability_floor,
            "inventory_scale_units": self.inventory_scale_units,
            "public_volatility_scale_ticks": self.public_volatility_scale_ticks,
            "mean_quote_scale_ticks": self.mean_quote_scale_ticks,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.probability_floor >= 1.0:
            raise ValueError("probability_floor must be below one")
        for name in (
            "inventory_feature_index",
            "previous_action_feature_index",
            "mean_quote_feature_index",
            "public_volatility_feature_index",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


def build_audit_batch(
    obs: np.ndarray,
    spec: TypedAuditSpec,
    always_on: bool = False,
) -> dict[str, np.ndarray]:
    """Construct shadow observations, validity flags, and response-set masks.

    ``response_mask`` is the typed response set and is empty on invalid rows.
    ``measurement_response_mask`` retains the fixed acceptable-action set on
    every row solely so callers can compute q_V and q_X against one common set.
    ``valid_mask`` remains the sole multiplier in the loss, making every
    invalid-row contribution exactly zero.
    """

    observations = np.asarray(obs, dtype=np.float32)
    if observations.ndim < 1 or observations.shape[-1] <= 0:
        raise ValueError(f"invalid typed-audit observation shape: {observations.shape}")
    if not np.all(np.isfinite(observations)):
        raise FloatingPointError("typed-audit observations contain non-finite values")
    feature_indices = (
        spec.inventory_feature_index,
        spec.previous_action_feature_index,
        spec.mean_quote_feature_index,
        spec.public_volatility_feature_index,
    )
    if max(feature_indices) >= observations.shape[-1]:
        raise ValueError(
            "typed-audit feature index exceeds observation width: "
            f"indices={feature_indices}, width={observations.shape[-1]}"
        )

    inventory_units = (
        observations[..., spec.inventory_feature_index]
        * spec.inventory_scale_units
    )
    volatility_ticks = (
        observations[..., spec.public_volatility_feature_index]
        * spec.public_volatility_scale_ticks
    )
    if always_on:
        valid_mask = np.ones(observations.shape[:-1], dtype=bool)
    else:
        valid_mask = (
            (np.abs(inventory_units) <= spec.absolute_inventory_max_units)
            & (volatility_ticks <= spec.public_volatility_max_ticks)
            & (volatility_ticks >= 0.0)
        )

    shadow_obs = observations.copy()
    previous_action = (
        observations[..., spec.previous_action_feature_index]
        * float(spec.n_actions - 1)
    )
    shadow_obs[..., spec.previous_action_feature_index] = (
        float(spec.shadow_action_index) / float(spec.n_actions - 1)
    )
    mean_quote_delta = (
        float(spec.shadow_action_index) - previous_action
    ) / float(spec.n_agents * spec.mean_quote_scale_ticks)
    shadow_obs[..., spec.mean_quote_feature_index] += mean_quote_delta.astype(
        np.float32, copy=False
    )

    response_template = np.zeros(spec.n_actions, dtype=bool)
    response_template[list(spec.acceptable_action_indices)] = True
    measurement_response_mask = np.broadcast_to(
        response_template, observations.shape[:-1] + (spec.n_actions,)
    ).copy()
    response_mask = measurement_response_mask & valid_mask[..., None]
    if not np.all(np.isfinite(shadow_obs)):
        raise FloatingPointError("typed-audit shadow observations are non-finite")
    return {
        "shadow_obs": shadow_obs.astype(np.float32, copy=False),
        "valid_mask": valid_mask,
        "response_mask": response_mask,
        "measurement_response_mask": measurement_response_mask,
    }


def _normalized_masks(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray,
    response_mask: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=probabilities.device)
    response = torch.as_tensor(
        response_mask, dtype=torch.bool, device=probabilities.device
    )
    if valid.ndim == probabilities.ndim and valid.shape[-1] == 1:
        valid = valid.squeeze(-1)
    if tuple(valid.shape) != tuple(probabilities.shape[:-1]):
        raise ValueError(
            f"valid-mask shape {tuple(valid.shape)} does not match probabilities "
            f"{tuple(probabilities.shape)}"
        )
    if tuple(response.shape) != tuple(probabilities.shape):
        raise ValueError(
            f"response-mask shape {tuple(response.shape)} does not match probabilities "
            f"{tuple(probabilities.shape)}"
        )
    if torch.any(valid & ~response.any(dim=-1)):
        raise ValueError("a valid audit row has an empty response set")
    return valid, response


def set_audit_loss_from_probs(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor | np.ndarray,
    response_mask: torch.Tensor | np.ndarray,
    probability_floor: float = 1e-12,
) -> torch.Tensor:
    """Return the complete-batch mean set loss with exact-zero invalid rows."""

    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    if probabilities.ndim < 2 or probabilities.shape[-1] <= 1:
        raise ValueError(
            f"invalid action-probability shape: {tuple(probabilities.shape)}"
        )
    if not np.isfinite(probability_floor) or not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must be finite and in (0, 1)")
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("audit action probabilities are non-finite")
    if torch.any(probabilities < 0.0):
        raise FloatingPointError("audit action probabilities are negative")
    row_sums = probabilities.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise FloatingPointError("audit action probabilities do not sum to one")

    valid, response = _normalized_masks(probabilities, valid_mask, response_mask)
    response_mass = (probabilities * response.to(probabilities.dtype)).sum(dim=-1)
    if torch.any(valid & (response_mass <= 0.0)):
        raise FloatingPointError("a valid audit row has zero response-set mass")
    valid_losses = -torch.log(response_mass.clamp_min(float(probability_floor)))
    per_row_loss = torch.where(valid, valid_losses, torch.zeros_like(valid_losses))
    loss = per_row_loss.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("typed-audit set loss is non-finite")
    return loss


def _audit_probabilities(
    learner: Any,
    original_obs: np.ndarray,
    shadow_obs: np.ndarray,
    rnn_states: np.ndarray,
    masks: np.ndarray,
    available_actions: np.ndarray | None,
) -> torch.Tensor:
    actor = learner.policy.actor
    device = learner.device
    original_features = actor.base(
        torch.as_tensor(original_obs, dtype=torch.float32, device=device)
    )
    shadow_features = actor.base(
        torch.as_tensor(shadow_obs, dtype=torch.float32, device=device)
    )
    if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
        hidden = torch.as_tensor(rnn_states, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=device)
        sequence_count = hidden.shape[0]
        if sequence_count <= 0 or shadow_features.shape[0] % sequence_count != 0:
            raise ValueError("audit recurrent minibatch cannot be split into sequences")
        shadow_outputs = []
        for start in range(0, shadow_features.shape[0], sequence_count):
            stop = start + sequence_count
            prior_hidden = hidden
            _, hidden = actor.rnn(
                original_features[start:stop], prior_hidden, mask_tensor[start:stop]
            )
            shadow_output, _ = actor.rnn(
                shadow_features[start:stop], prior_hidden, mask_tensor[start:stop]
            )
            shadow_outputs.append(shadow_output)
        features = torch.cat(shadow_outputs, dim=0)
    else:
        features = shadow_features
    available = None
    if available_actions is not None:
        available = torch.as_tensor(
            available_actions, dtype=torch.float32, device=device
        )
    probabilities = actor.act.get_probs(features, available)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("typed-audit actor produced non-finite probabilities")
    return probabilities


def _gradient_norm(parameters: Any) -> torch.Tensor:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        term = parameter.grad.detach().pow(2).sum()
        squared = term if squared is None else squared + term
    return torch.tensor(0.0) if squared is None else squared.sqrt()


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    result = float(value)
    if not np.isfinite(result):
        raise FloatingPointError(f"non-finite typed-audit diagnostic: {result}")
    return result


def official_update_with_set_audit(
    learner: Any,
    buffer: Any,
    update_seed: int,
    spec: TypedAuditSpec,
    always_on: bool = False,
) -> dict[str, float]:
    """Run pinned official PPO with a full-batch set loss on each actor minibatch."""

    if learner.config.n_actions != spec.n_actions:
        raise ValueError("typed-audit and learner action counts differ")
    if learner.config.n_agents != spec.n_agents:
        raise ValueError("typed-audit and learner agent counts differ")
    actor_ids = {id(parameter) for parameter in learner.policy.actor.parameters()}
    critic_ids = {id(parameter) for parameter in learner.policy.critic.parameters()}
    if actor_ids & critic_ids:
        raise RuntimeError("typed-audit update requires disjoint actor and critic parameters")

    next_values = learner.next_values(
        buffer.share_obs[-1, 0],
        buffer.rnn_states_critic[-1, 0],
        buffer.masks[-1, 0],
    )
    buffer.compute_returns(next_values, learner.trainer.value_normalizer)
    if not np.all(np.isfinite(buffer.returns)):
        raise FloatingPointError("official buffer produced non-finite financial returns")

    torch.manual_seed(int(update_seed))
    np.random.seed(int(update_seed) % (2**32 - 1))
    trainer = learner.trainer
    policy = learner.policy
    trainer.prep_training()

    if trainer._use_popart or trainer._use_valuenorm:
        advantages = buffer.returns[:-1] - trainer.value_normalizer.denormalize(
            buffer.value_preds[:-1]
        )
    else:
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
    advantages_copy = advantages.copy()
    advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
    mean_advantages = np.nanmean(advantages_copy)
    std_advantages = np.nanstd(advantages_copy)
    if not np.isfinite(mean_advantages) or not np.isfinite(std_advantages):
        raise FloatingPointError("official financial advantages are non-finite")
    advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    canonical_batch = build_audit_batch(buffer.obs[:-1], spec, always_on=always_on)
    canonical_active = np.asarray(buffer.active_masks[:-1]).squeeze(-1) > 0.0
    canonical_valid = canonical_batch["valid_mask"] & canonical_active
    audit_total_rows = int(canonical_valid.size)
    audit_valid_rows = int(np.count_nonzero(canonical_valid))

    totals = {
        "value_loss": 0.0,
        "policy_loss": 0.0,
        "dist_entropy": 0.0,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
        "ratio": 0.0,
        "audit_loss": 0.0,
        "audit_response_mass": 0.0,
    }
    updates = 0
    response_mass_updates = 0
    valid_rows_seen = 0
    zero_valid_minibatches = 0
    critic_grad_max_abs = 0.0
    invalid_loss_max_abs = 0.0

    for _ in range(trainer.ppo_epoch):
        if trainer._use_recurrent_policy:
            generator = buffer.recurrent_generator(
                advantages, trainer.num_mini_batch, trainer.data_chunk_length
            )
        elif trainer._use_naive_recurrent:
            generator = buffer.naive_recurrent_generator(
                advantages, trainer.num_mini_batch
            )
        else:
            generator = buffer.feed_forward_generator(
                advantages, trainer.num_mini_batch
            )

        for sample in generator:
            (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                *_ignored,
            ) = sample
            device = learner.device
            old_log_probs = torch.as_tensor(
                old_action_log_probs_batch, dtype=torch.float32, device=device
            )
            advantage_targets = torch.as_tensor(
                adv_targ, dtype=torch.float32, device=device
            )
            value_predictions = torch.as_tensor(
                value_preds_batch, dtype=torch.float32, device=device
            )
            financial_returns = torch.as_tensor(
                return_batch, dtype=torch.float32, device=device
            )
            active_masks = torch.as_tensor(
                active_masks_batch, dtype=torch.float32, device=device
            )
            if not all(
                torch.isfinite(tensor).all()
                for tensor in (
                    old_log_probs,
                    advantage_targets,
                    value_predictions,
                    financial_returns,
                    active_masks,
                )
            ):
                raise FloatingPointError("official PPO minibatch contains non-finite values")
            if active_masks.sum() <= 0.0:
                raise ValueError("official PPO minibatch has no active rows")

            values, action_log_probs, dist_entropy = policy.evaluate_actions(
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )
            importance_weights = torch.exp(action_log_probs - old_log_probs)
            surrogate_one = importance_weights * advantage_targets
            surrogate_two = torch.clamp(
                importance_weights,
                1.0 - trainer.clip_param,
                1.0 + trainer.clip_param,
            ) * advantage_targets
            if trainer._use_policy_active_masks:
                policy_action_loss = (
                    -torch.sum(
                        torch.min(surrogate_one, surrogate_two),
                        dim=-1,
                        keepdim=True,
                    )
                    * active_masks
                ).sum() / active_masks.sum()
            else:
                policy_action_loss = -torch.sum(
                    torch.min(surrogate_one, surrogate_two), dim=-1, keepdim=True
                ).mean()
            policy_loss = policy_action_loss

            audit_batch = build_audit_batch(
                np.asarray(obs_batch, dtype=np.float32), spec, always_on=always_on
            )
            minibatch_valid = audit_batch["valid_mask"] & (
                np.asarray(active_masks_batch).reshape(-1) > 0.0
            )
            audit_probabilities = _audit_probabilities(
                learner,
                np.asarray(obs_batch, dtype=np.float32),
                audit_batch["shadow_obs"],
                rnn_states_batch,
                masks_batch,
                available_actions_batch,
            )
            audit_loss = set_audit_loss_from_probs(
                audit_probabilities,
                minibatch_valid,
                audit_batch["response_mask"],
                probability_floor=spec.probability_floor,
            )
            valid_tensor, response_tensor = _normalized_masks(
                audit_probabilities,
                minibatch_valid,
                audit_batch["response_mask"],
            )
            response_mass = (
                audit_probabilities * response_tensor.to(audit_probabilities.dtype)
            ).sum(dim=-1)
            per_row = torch.where(
                valid_tensor,
                -torch.log(response_mass.clamp_min(spec.probability_floor)),
                torch.zeros_like(response_mass),
            )
            if torch.any(~valid_tensor):
                invalid_loss_max_abs = max(
                    invalid_loss_max_abs,
                    _scalar(per_row[~valid_tensor].abs().max()),
                )
            minibatch_valid_rows = int(valid_tensor.sum().detach().cpu().item())
            valid_rows_seen += minibatch_valid_rows
            if minibatch_valid_rows:
                totals["audit_response_mass"] += _scalar(
                    response_mass[valid_tensor].mean()
                )
                response_mass_updates += 1
            else:
                zero_valid_minibatches += 1

            policy.actor_optimizer.zero_grad()
            policy.critic_optimizer.zero_grad(set_to_none=True)
            actor_objective = (
                policy_loss
                - dist_entropy * trainer.entropy_coef
                + spec.actor_coefficient * audit_loss
            )
            if not torch.isfinite(actor_objective):
                raise FloatingPointError("typed-audit actor objective is non-finite")
            actor_objective.backward()
            for parameter in policy.critic.parameters():
                if parameter.grad is not None:
                    critic_grad_max_abs = max(
                        critic_grad_max_abs,
                        _scalar(parameter.grad.detach().abs().max()),
                    )
            if critic_grad_max_abs != 0.0:
                raise RuntimeError("actor update created a critic gradient")
            if trainer._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    policy.actor.parameters(), trainer.max_grad_norm
                )
            else:
                actor_grad_norm = _gradient_norm(policy.actor.parameters())
            policy.actor_optimizer.step()

            value_loss = trainer.cal_value_loss(
                values, value_predictions, financial_returns, active_masks
            )
            policy.critic_optimizer.zero_grad()
            (value_loss * trainer.value_loss_coef).backward()
            if trainer._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    policy.critic.parameters(), trainer.max_grad_norm
                )
            else:
                critic_grad_norm = _gradient_norm(policy.critic.parameters())
            policy.critic_optimizer.step()

            if not all(
                torch.isfinite(tensor).all()
                for tensor in (
                    values,
                    action_log_probs,
                    dist_entropy,
                    importance_weights,
                    policy_loss,
                    audit_loss,
                    value_loss,
                )
            ):
                raise FloatingPointError("typed-audit PPO update produced non-finite values")
            totals["value_loss"] += _scalar(value_loss)
            totals["policy_loss"] += _scalar(policy_loss)
            totals["dist_entropy"] += _scalar(dist_entropy)
            totals["actor_grad_norm"] += _scalar(actor_grad_norm)
            totals["critic_grad_norm"] += _scalar(critic_grad_norm)
            totals["ratio"] += _scalar(importance_weights.mean())
            totals["audit_loss"] += _scalar(audit_loss)
            updates += 1

    expected_updates = trainer.ppo_epoch * trainer.num_mini_batch
    if updates != expected_updates or updates <= 0:
        raise RuntimeError(
            f"typed-audit PPO update count mismatch: {updates} != {expected_updates}"
        )
    buffer.after_update()
    result = {
        key: value / updates
        for key, value in totals.items()
        if key != "audit_response_mass"
    }
    result["audit_response_mass"] = (
        totals["audit_response_mass"] / response_mass_updates
        if response_mass_updates
        else 0.0
    )
    result.update(
        {
            "audit_weighted_loss": result["audit_loss"]
            * spec.actor_coefficient,
            "audit_actor_coefficient": float(spec.actor_coefficient),
            "audit_total_rows": float(audit_total_rows),
            "audit_valid_rows": float(audit_valid_rows),
            "audit_valid_rate": (
                float(audit_valid_rows) / float(audit_total_rows)
                if audit_total_rows
                else 0.0
            ),
            "audit_valid_rows_seen": float(valid_rows_seen),
            "audit_zero_valid_minibatches": float(zero_valid_minibatches),
            "audit_invalid_loss_max_abs": float(invalid_loss_max_abs),
            "audit_critic_grad_max_abs": float(critic_grad_max_abs),
        }
    )
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"non-finite typed-audit update: {result}")
    return result
