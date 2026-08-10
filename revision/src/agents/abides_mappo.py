"""MAPPO rollout helpers for the host-to-ABIDES Docker bridge."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.agents.mappo import MAPPOConfig, MAPPOTrainer
from src.envs.abides_docker_env import ABIDESDockerDealerEnv

GateDecision = Callable[[Mapping[str, Any], np.ndarray, int], bool]
ContractDecision = Callable[[Mapping[str, Any], int], float | Mapping[str, Any]]


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def actor_observations(
    state: Mapping[str, Any], gate_signal: float = 0.0
) -> np.ndarray:
    base = np.asarray(state["observations"], dtype=np.float32)
    if base.shape != (10, 6):
        raise ValueError(
            f"expected ABIDES observations with shape (10, 6), got {base.shape}"
        )
    signal = np.full((base.shape[0], 1), float(gate_signal), dtype=np.float32)
    return np.concatenate([base, signal], axis=1)


def centralized_critic_observations(
    state: Mapping[str, Any], observations: np.ndarray, episode_steps: int
) -> np.ndarray:
    base = observations[:, :-1]
    info = state["info"]
    global_features = np.concatenate(
        [
            np.mean(base, axis=0),
            np.std(base, axis=0),
            np.asarray(
                [
                    float(info.get("avg_half_spread", 1.0)) / 7.0,
                    float(info.get("public_volatility", 0.0)) / 4.0,
                    float(info.get("completed_steps", 0)) / max(episode_steps, 1),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    tiled = np.repeat(global_features[None, :], observations.shape[0], axis=0)
    return np.concatenate([observations, tiled], axis=1).astype(np.float32)


def make_abides_trainer(
    seed: int,
    episode_steps: int,
    device: str,
    entropy_coef: float = 0.02,
    policy_mode: str = "multihead",
    policy_advantage_mode: str = "relative_immediate",
    learning_rate: float = 1e-3,
    adapter_risk_feature_index: int | None = 0,
) -> MAPPOTrainer:
    dummy_state = {
        "observations": [[0.0] * 6 for _ in range(10)],
        "info": {
            "avg_half_spread": 1.0,
            "public_volatility": 0.0,
            "completed_steps": 0,
        },
    }
    obs = actor_observations(dummy_state)
    critic = centralized_critic_observations(dummy_state, obs, episode_steps)
    return MAPPOTrainer(
        MAPPOConfig(
            obs_dim=obs.shape[1],
            critic_obs_dim=critic.shape[1],
            n_actions=7,
            n_agents=10,
            policy_mode=policy_mode,
            policy_advantage_mode=policy_advantage_mode,
            hidden_dim=64,
            learning_rate=learning_rate,
            gamma=0.96,
            gae_lambda=0.95,
            clip_ratio=0.20,
            entropy_coef=entropy_coef,
            value_coef=0.25,
            max_grad_norm=0.50,
            update_epochs=4,
            minibatch_size=512,
            adapter_risk_feature_index=adapter_risk_feature_index,
            seed=seed,
            device=device,
        )
    )


def rollout_episode(
    env: ABIDESDockerDealerEnv,
    trainer: MAPPOTrainer,
    seed: int,
    episode_steps: int,
    train: bool,
    gate_schedule: Sequence[bool] | None = None,
    gate_decision: GateDecision | None = None,
    announced_signal_schedule: Sequence[float] | None = None,
    announced_signal_decision: ContractDecision | None = None,
    routing_enabled: bool = True,
    reward_scale: float = 10.0,
    action_namespace: int = 73_117,
    deterministic: bool = False,
    defer_update: bool = False,
    routing_mechanism: Mapping[str, Any] | None = None,
    dealer_signal_schedule: Sequence[float] | None = None,
    dealer_signal_decision: ContractDecision | None = None,
    signal_audit_schedule: Sequence[Mapping[str, Any]] | None = None,
    routing_mechanism_schedule: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if gate_schedule is not None and gate_decision is not None:
        raise ValueError("provide either gate_schedule or gate_decision, not both")
    if (
        announced_signal_schedule is not None or announced_signal_decision is not None
    ) and (gate_schedule is not None or gate_decision is not None):
        raise ValueError("announced signals cannot be combined with post-action gates")
    if announced_signal_schedule is not None and announced_signal_decision is not None:
        raise ValueError(
            "provide either announced signal schedule or decision, not both"
        )
    if dealer_signal_schedule is not None and dealer_signal_decision is not None:
        raise ValueError("provide either dealer signal schedule or decision, not both")
    if gate_schedule is not None and len(gate_schedule) != episode_steps:
        raise ValueError("gate schedule length must match episode_steps")
    if (
        announced_signal_schedule is not None
        and len(announced_signal_schedule) != episode_steps
    ):
        raise ValueError("announced signal schedule length must match episode_steps")
    for name, schedule in (
        ("dealer signal", dealer_signal_schedule),
        ("signal audit", signal_audit_schedule),
        ("routing mechanism", routing_mechanism_schedule),
    ):
        if schedule is not None and len(schedule) != episode_steps:
            raise ValueError(f"{name} schedule length must match episode_steps")
    if routing_mechanism is not None and routing_mechanism_schedule is not None:
        raise ValueError(
            "provide either a static routing mechanism or a mechanism schedule"
        )

    def contract_value(
        value: float | Mapping[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        if isinstance(value, Mapping):
            audit = dict(value)
            raw = audit.get("applied_signal", audit.get("strength"))
            if raw is None:
                raise ValueError("contract decision mapping has no applied signal")
            return float(np.clip(float(raw), 0.0, 1.0)), audit
        return float(np.clip(float(value), 0.0, 1.0)), {}

    state = env.reset(seed=seed)
    obs_rows = []
    critic_rows = []
    action_rows = []
    logp_rows = []
    value_rows = []
    reward_rows = []
    records = []
    for step in range(episode_steps):
        announced_strength: float | None = None
        signal_audit: dict[str, Any] = (
            {}
            if signal_audit_schedule is None
            else dict(signal_audit_schedule[step])
        )
        if announced_signal_schedule is not None:
            announced_strength = float(
                np.clip(float(announced_signal_schedule[step]), 0.0, 1.0)
            )
        elif announced_signal_decision is not None:
            announced_strength, decision_audit = contract_value(
                announced_signal_decision(state, step)
            )
            signal_audit = {**decision_audit, **signal_audit}
        dealer_strength = announced_strength
        if dealer_signal_schedule is not None:
            dealer_strength = float(
                np.clip(float(dealer_signal_schedule[step]), 0.0, 1.0)
            )
        elif dealer_signal_decision is not None:
            dealer_strength, dealer_audit = contract_value(
                dealer_signal_decision(state, step)
            )
            signal_audit = {
                **signal_audit,
                "dealer_signal_decision": dealer_audit,
            }
        obs = actor_observations(
            state,
            gate_signal=0.0 if dealer_strength is None else dealer_strength,
        )
        critic = centralized_critic_observations(state, obs, episode_steps)
        action_uniforms = None
        if not deterministic:
            action_uniforms = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(step), int(action_namespace)])
            ).random(10)
        action_uniform_digest = canonical_digest(
            [] if action_uniforms is None else action_uniforms.tolist()
        )
        pre_state_digest = canonical_digest(
            {"observations": state["observations"], "info": state["info"]}
        )
        actor_observation_digest = canonical_digest(obs.tolist())
        critic_observation_digest = canonical_digest(critic.tolist())
        actions, logp, values = trainer.act(
            obs,
            critic,
            deterministic=deterministic,
            action_uniforms=action_uniforms,
        )
        if announced_strength is not None:
            active = announced_strength > 0.0
        elif gate_schedule is not None:
            active = bool(gate_schedule[step])
        elif gate_decision is not None:
            active = bool(gate_decision(state, actions, step))
        else:
            active = False
        step_routing_mechanism = (
            routing_mechanism
            if routing_mechanism_schedule is None
            else routing_mechanism_schedule[step]
        )
        next_state = env.step(
            actions,
            gate_active=active,
            routing_enabled=routing_enabled,
            routing_strength=announced_strength,
            routing_mechanism=step_routing_mechanism,
        )
        rewards = np.asarray(next_state["rewards"], dtype=float) / reward_scale
        obs_rows.append(obs)
        critic_rows.append(critic)
        action_rows.append(actions.astype(np.int64))
        logp_rows.append(logp.astype(float))
        value_rows.append(values.astype(float))
        reward_rows.append(rewards)
        records.append(
            {
                "step": step,
                "actions": actions.tolist(),
                "gate_active": active,
                "gate_signal": (
                    0.0 if announced_strength is None else announced_strength
                ),
                "platform_applied_signal": (
                    0.0 if announced_strength is None else announced_strength
                ),
                "dealer_observed_signal": (
                    0.0 if dealer_strength is None else dealer_strength
                ),
                "signal_audit": signal_audit,
                "pre_state_digest": pre_state_digest,
                "actor_observation_digest": actor_observation_digest,
                "critic_observation_digest": critic_observation_digest,
                "action_uniform_digest": action_uniform_digest,
                "pre_info": dict(state["info"]),
                "pre_observations": state["observations"],
                "info": dict(next_state["info"]),
                "post_observations": next_state["observations"],
                "post_state_digest": canonical_digest(
                    {
                        "observations": next_state["observations"],
                        "info": next_state["info"],
                    }
                ),
                "rewards_unscaled": list(next_state["rewards"]),
                "post_agents": list(next_state["agents"]),
                "mean_reward_unscaled": float(np.mean(next_state["rewards"])),
            }
        )
        state = next_state
    if not state["done"]:
        raise RuntimeError("ABIDES episode did not finish at configured horizon")
    env.close_episode()

    trajectory = {
        "obs": np.stack(obs_rows),
        "critic_obs": np.stack(critic_rows),
        "actions": np.stack(action_rows),
        "old_logp": np.stack(logp_rows),
        "rewards": np.stack(reward_rows),
        "values": np.stack(value_rows),
    }
    update = {}
    if train and not defer_update:
        update = trainer.update_episode(
            trajectory["obs"],
            trajectory["critic_obs"],
            trajectory["actions"],
            trajectory["old_logp"],
            trajectory["rewards"],
            trajectory["values"],
        )
        if not all(np.isfinite(float(value)) for value in update.values()):
            raise FloatingPointError(f"non-finite MAPPO update: {update}")
    return {
        "update": update,
        "trajectory": trajectory if train else None,
        "records": records,
        "mean_action": float(np.mean(action_rows)),
        "mean_half_spread": float(np.mean(np.asarray(action_rows) + 1)),
        "mean_reward": float(
            np.mean([record["mean_reward_unscaled"] for record in records])
        ),
        "activation_rate": float(
            np.mean([record["gate_active"] for record in records])
        ),
        "mean_gate_signal": float(
            np.mean([record["gate_signal"] for record in records])
        ),
        "mean_dealer_observed_signal": float(
            np.mean([record["dealer_observed_signal"] for record in records])
        ),
    }
