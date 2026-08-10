"""Thin, auditable wrapper around the pinned marlbenchmark/on-policy MAPPO."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


OFFICIAL_COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"


class Box:
    """Minimal shape adapter required by the pinned official implementation."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class Discrete:
    """Minimal discrete-action adapter required by the official ACTLayer."""

    def __init__(self, n: int):
        self.n = int(n)


@dataclass(frozen=True)
class OfficialMAPPOConfig:
    source_root: str
    seed: int
    episode_steps: int = 100
    n_agents: int = 10
    obs_dim: int = 7
    centralized_obs_dim: int = 22
    n_actions: int = 7
    centralized_value: bool = True
    hidden_size: int = 64
    recurrent_n: int = 1
    data_chunk_length: int = 10
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_coefficient: float = 0.01
    ppo_epoch: int = 10
    num_mini_batch: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_parameter: float = 0.2
    value_loss_coefficient: float = 1.0
    max_gradient_norm: float = 10.0
    reward_scale: float = 100.0
    device: str = "cpu"

    @property
    def critic_obs_dim(self) -> int:
        return self.centralized_obs_dim if self.centralized_value else self.obs_dim


def verify_official_source(source_root: str | Path) -> dict[str, str]:
    root = Path(source_root).expanduser().resolve()
    if not (root / "onpolicy").is_dir() or not (root / ".git").is_dir():
        raise RuntimeError(f"invalid official on-policy checkout: {root}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official on-policy commit mismatch: {commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise RuntimeError("official on-policy checkout is dirty")
    return {"path": str(root), "commit": commit}


def _load_official_classes(source_root: Path) -> tuple[type, type, type]:
    root_text = str(source_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    package_root = source_root / "onpolicy"
    existing = sys.modules.get("onpolicy")
    if existing is None:
        package = types.ModuleType("onpolicy")
        package.__file__ = str(package_root / "__init__.py")
        package.__path__ = [str(package_root)]
        package.__package__ = "onpolicy"
        sys.modules["onpolicy"] = package
    elif str(package_root) not in list(getattr(existing, "__path__", [])):
        raise RuntimeError("another onpolicy source is already imported")
    policy_module = importlib.import_module(
        "onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy"
    )
    trainer_module = importlib.import_module("onpolicy.algorithms.r_mappo.r_mappo")
    buffer_module = importlib.import_module("onpolicy.utils.shared_buffer")
    return (
        policy_module.R_MAPPOPolicy,
        trainer_module.R_MAPPO,
        buffer_module.SharedReplayBuffer,
    )


def _official_args(config: OfficialMAPPOConfig) -> Namespace:
    return Namespace(
        algorithm_name="rmappo",
        episode_length=config.episode_steps,
        n_rollout_threads=1,
        hidden_size=config.hidden_size,
        recurrent_N=config.recurrent_n,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        use_gae=True,
        use_popart=False,
        use_valuenorm=True,
        use_proper_time_limits=False,
        use_recurrent_policy=True,
        use_naive_recurrent_policy=False,
        data_chunk_length=config.data_chunk_length,
        lr=config.learning_rate,
        critic_lr=config.critic_learning_rate,
        opti_eps=1e-5,
        weight_decay=0.0,
        layer_N=1,
        stacked_frames=1,
        use_stacked_frames=False,
        use_ReLU=True,
        use_feature_normalization=True,
        use_orthogonal=True,
        gain=0.01,
        use_policy_active_masks=True,
        ppo_epoch=config.ppo_epoch,
        num_mini_batch=config.num_mini_batch,
        clip_param=config.clip_parameter,
        value_loss_coef=config.value_loss_coefficient,
        entropy_coef=config.entropy_coefficient,
        use_max_grad_norm=True,
        max_grad_norm=config.max_gradient_norm,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        huber_delta=10.0,
        use_value_active_masks=True,
    )


class OfficialMAPPO:
    """Official actor, critic, PPO trainer, and shared replay buffer adapter."""

    def __init__(self, config: OfficialMAPPOConfig):
        self.config = config
        source = verify_official_source(config.source_root)
        self.source_commit = source["commit"]
        source_root = Path(source["path"])
        policy_type, trainer_type, buffer_type = _load_official_classes(source_root)
        self._buffer_type = buffer_type
        self.args = _official_args(config)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device(config.device)
        self.obs_space = Box((config.obs_dim,))
        self.share_obs_space = Box((config.critic_obs_dim,))
        self.action_space = Discrete(config.n_actions)
        self.policy = policy_type(
            self.args,
            self.obs_space,
            self.share_obs_space,
            self.action_space,
            self.device,
        )
        self.trainer = trainer_type(self.args, self.policy, self.device)

    def new_buffer(self, obs: np.ndarray, critic_obs: np.ndarray) -> Any:
        obs = np.asarray(obs, dtype=np.float32)
        critic_obs = np.asarray(critic_obs, dtype=np.float32)
        self._validate_observations(obs, critic_obs)
        buffer = self._buffer_type(
            self.args,
            self.config.n_agents,
            self.obs_space,
            self.share_obs_space,
            self.action_space,
        )
        buffer.obs[0, 0] = obs
        buffer.share_obs[0, 0] = critic_obs
        return buffer

    def initial_recurrent_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (
            self.config.n_agents,
            self.config.recurrent_n,
            self.config.hidden_size,
        )
        actor = np.zeros(shape, dtype=np.float32)
        critic = np.zeros(shape, dtype=np.float32)
        masks = np.ones((self.config.n_agents, 1), dtype=np.float32)
        return actor, critic, masks

    def _validate_observations(
        self, obs: np.ndarray, critic_obs: np.ndarray
    ) -> None:
        if obs.shape != (self.config.n_agents, self.config.obs_dim):
            raise ValueError(f"invalid official actor observation shape: {obs.shape}")
        if critic_obs.shape != (
            self.config.n_agents,
            self.config.critic_obs_dim,
        ):
            raise ValueError(
                f"invalid official critic observation shape: {critic_obs.shape}"
            )

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        critic_obs: np.ndarray,
        rnn_actor: np.ndarray,
        rnn_critic: np.ndarray,
        masks: np.ndarray,
        action_uniforms: np.ndarray | None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        critic_obs = np.asarray(critic_obs, dtype=np.float32)
        self._validate_observations(obs, critic_obs)
        self.trainer.prep_rollout()
        actor = self.policy.actor
        actor_features = actor.base(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        )
        rnn_actor_tensor = torch.as_tensor(
            rnn_actor, dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        actor_features, rnn_actor_next = actor.rnn(
            actor_features, rnn_actor_tensor, mask_tensor
        )
        distribution = actor.act.action_out(actor_features)
        if deterministic:
            if action_uniforms is not None:
                raise ValueError("deterministic actions cannot receive action uniforms")
            actions = distribution.mode()
        else:
            if action_uniforms is None:
                raise ValueError("stochastic official actions require explicit uniforms")
            uniforms = torch.as_tensor(
                action_uniforms, dtype=distribution.probs.dtype, device=self.device
            )
            if tuple(uniforms.shape) != (self.config.n_agents,):
                raise ValueError(f"invalid action uniform shape: {uniforms.shape}")
            cumulative = torch.cumsum(distribution.probs, dim=-1)
            actions = torch.sum(
                uniforms[:, None] > cumulative, dim=-1, keepdim=True
            ).clamp(max=self.config.n_actions - 1)
        action_log_probs = distribution.log_probs(actions)
        values, rnn_critic_next = self.policy.critic(
            critic_obs,
            rnn_critic,
            masks,
        )
        return (
            actions.cpu().numpy().astype(np.int64),
            action_log_probs.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
            rnn_actor_next.cpu().numpy().astype(np.float32),
            rnn_critic_next.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def action_probabilities(
        self,
        obs: np.ndarray,
        rnn_actor: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Return actor probabilities without sampling or changing learner state."""

        obs = np.asarray(obs, dtype=np.float32)
        rnn_actor = np.asarray(rnn_actor, dtype=np.float32)
        masks = np.asarray(masks, dtype=np.float32)
        if obs.shape != (self.config.n_agents, self.config.obs_dim):
            raise ValueError(f"invalid official actor observation shape: {obs.shape}")
        expected_rnn_shape = (
            self.config.n_agents,
            self.config.recurrent_n,
            self.config.hidden_size,
        )
        if rnn_actor.shape != expected_rnn_shape:
            raise ValueError(f"invalid official actor RNN-state shape: {rnn_actor.shape}")
        if masks.shape != (self.config.n_agents, 1):
            raise ValueError(f"invalid official actor mask shape: {masks.shape}")
        if not (
            np.all(np.isfinite(obs))
            and np.all(np.isfinite(rnn_actor))
            and np.all(np.isfinite(masks))
        ):
            raise FloatingPointError("non-finite input to official action probabilities")

        self.trainer.prep_rollout()
        actor = self.policy.actor
        actor_features = actor.base(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        )
        actor_features, _ = actor.rnn(
            actor_features,
            torch.as_tensor(rnn_actor, dtype=torch.float32, device=self.device),
            torch.as_tensor(masks, dtype=torch.float32, device=self.device),
        )
        probabilities = actor.act.get_probs(actor_features)
        result = probabilities.detach().cpu().numpy().astype(np.float32)
        expected_probability_shape = (
            self.config.n_agents,
            self.config.n_actions,
        )
        if result.shape != expected_probability_shape:
            raise RuntimeError(
                "official actor returned invalid probability shape: "
                f"{result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("official actor returned non-finite probabilities")
        if np.any(result < 0.0) or not np.allclose(
            result.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise FloatingPointError("official actor returned invalid probabilities")
        return result

    def insert(
        self,
        buffer: Any,
        next_obs: np.ndarray,
        next_critic_obs: np.ndarray,
        rnn_actor: np.ndarray,
        rnn_critic: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        buffer.insert(
            np.asarray(next_critic_obs, dtype=np.float32)[None, ...],
            np.asarray(next_obs, dtype=np.float32)[None, ...],
            np.asarray(rnn_actor, dtype=np.float32)[None, ...],
            np.asarray(rnn_critic, dtype=np.float32)[None, ...],
            np.asarray(actions, dtype=np.float32)[None, ...],
            np.asarray(action_log_probs, dtype=np.float32)[None, ...],
            np.asarray(values, dtype=np.float32)[None, ...],
            np.asarray(rewards, dtype=np.float32)[None, ..., None],
            np.asarray(masks, dtype=np.float32)[None, ...],
        )

    @torch.no_grad()
    def next_values(
        self, critic_obs: np.ndarray, rnn_critic: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        self.trainer.prep_rollout()
        values = self.policy.get_values(critic_obs, rnn_critic, masks)
        return values.cpu().numpy().astype(np.float32)[None, ...]

    def update(self, buffer: Any, update_seed: int) -> dict[str, float]:
        next_values = self.next_values(
            buffer.share_obs[-1, 0],
            buffer.rnn_states_critic[-1, 0],
            buffer.masks[-1, 0],
        )
        buffer.compute_returns(next_values, self.trainer.value_normalizer)
        torch.manual_seed(int(update_seed))
        np.random.seed(int(update_seed) % (2**32 - 1))
        self.trainer.prep_training()
        result = self.trainer.train(buffer)
        buffer.after_update()
        normalized = {
            key: (
                float(value.detach().cpu().item())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
            for key, value in result.items()
        }
        if not all(np.isfinite(value) for value in normalized.values()):
            raise FloatingPointError(f"non-finite official MAPPO update: {normalized}")
        return normalized

    def payload(self) -> dict[str, Any]:
        value_normalizer = self.trainer.value_normalizer
        return {
            "official_commit": self.source_commit,
            "config": asdict(self.config),
            "actor": self.policy.actor.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "actor_optimizer": self.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.policy.critic_optimizer.state_dict(),
            "value_normalizer": (
                None if value_normalizer is None else value_normalizer.state_dict()
            ),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OfficialMAPPO":
        if payload.get("official_commit") != OFFICIAL_COMMIT:
            raise RuntimeError("official MAPPO payload commit mismatch")
        instance = cls(OfficialMAPPOConfig(**payload["config"]))
        instance.policy.actor.load_state_dict(payload["actor"])
        instance.policy.critic.load_state_dict(payload["critic"])
        instance.policy.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        instance.policy.critic_optimizer.load_state_dict(
            payload["critic_optimizer"]
        )
        if payload.get("value_normalizer") is not None:
            instance.trainer.value_normalizer.load_state_dict(
                payload["value_normalizer"]
            )
        return instance

    def clone(self) -> "OfficialMAPPO":
        return self.from_payload(self.payload())

    def model_digest(self) -> str:
        digest = hashlib.sha256()
        for prefix, module in (
            ("actor", self.policy.actor),
            ("critic", self.policy.critic),
        ):
            for name, tensor in sorted(module.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()
