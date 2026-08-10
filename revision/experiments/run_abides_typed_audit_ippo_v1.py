#!/usr/bin/env python3
"""Frozen ABIDES typed-audit IPPO v2 response experiment.

This runner is deliberately separate from the historical routing experiments.
It consumes the frozen official-IPPO base checkpoints and compares four policy
update rules while keeping the live market path, action uniforms, financial
reward, and disabled-routing environment contract explicit in the artifacts.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.agents.abides_mappo import actor_observations, canonical_digest  # noqa: E402
from src.agents.official_on_policy_mappo import (  # noqa: E402
    OFFICIAL_COMMIT,
    OfficialMAPPO,
    verify_official_source,
)
from src.agents.typed_audit import (  # noqa: E402
    TypedAuditSpec,
    build_audit_batch,
    official_update_with_set_audit,
)
from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)
PROTOCOL_VERSION = "abides_typed_audit_ippo_v2"
EXPECTED_CONFIG_SHA256 = "92f250d4ad32cc97ee96ccafd21f023c58cf0cbd96e2227caee85e4d4833f950"
EXPECTED_PROTOCOL_SHA256 = "9bffb93817c2dc2fc4a90a1a4417045d765b183a194a3f84ed46473344b89e5a"
METHODS = ("vanilla_ippo", "action_cost", "always_on_actor", "full_ace")
FORMAL_POLICY_SEEDS = tuple(range(560000, 560010))
FORMAL_EVALUATION_SEEDS = tuple(range(701000000, 701000020))
FORMAL_RESPONSE_EPISODES = 96
SMOKE_RESPONSE_EPISODES = 2
REWARD_SCALE = 100.0
PRIMARY_METRICS = (
    "q_v",
    "q_x",
    "selectivity",
    "valid_audit_rate",
    "mean_per_dealer_financial_reward",
    "mean_per_dealer_gross_pnl",
    "mean_per_dealer_inventory_penalty",
    "mean_abs_inventory",
    "mean_cross_dealer_inventory_variance",
    "mean_quoted_half_spread",
    "effective_half_spread",
    "mean_fill_ratio",
    "total_volume",
    "total_customer_cost",
    "mean_quote_dispersion",
    "mean_best_quote_tie_share",
    "mean_pairwise_action_agreement",
    "mean_action_entropy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def code_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "agents" / "typed_audit.py",
        ROOT / "src" / "agents" / "official_on_policy_mappo.py",
        ROOT / "src" / "envs" / "abides_docker_env.py",
    )
    return {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in paths
    }


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_logbook(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now(), **event}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def resolve_docker_image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "images", "--quiet", "--no-trunc", image],
        check=True,
        capture_output=True,
        text=True,
    )
    image_ids = sorted(set(result.stdout.split()))
    if len(image_ids) != 1:
        raise RuntimeError(
            f"expected exactly one local Docker image for {image}, got {image_ids}"
        )
    subprocess.run(
        ["docker", "inspect", "--type", "image", image_ids[0]],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return image_ids[0]


def formal_worktree_violations(allowed_untracked_root: str | None) -> list[str]:
    del allowed_untracked_root
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [line for line in status if line]


def git_file_is_tracked(path: Path) -> bool:
    relative = str(path.resolve().relative_to(REPO))
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    def __init__(self, path: Path, fingerprint: str):
        self.path = path
        self.fingerprint = fingerprint
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
            same_host = existing.get("host") == socket.gethostname()
            active = same_host and pid_is_alive(int(existing.get("pid", -1)))
            if active:
                raise RuntimeError(
                    f"typed-audit run already active under pid {existing.get('pid')}"
                )
            if not same_host:
                raise RuntimeError(
                    "typed-audit run lock belongs to another host; inspect it manually"
                )
            self.path.unlink()
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_at": utc_now(),
                    "protocol_fingerprint": self.fingerprint,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_interval(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("t-interval values must be a nonempty finite vector")
    average = float(np.mean(data))
    if len(data) == 1:
        standard_error = 0.0
        low = high = average
    else:
        standard_error = float(np.std(data, ddof=1) / math.sqrt(len(data)))
        critical = T_CRITICAL_975.get(len(data) - 1)
        if critical is None:
            raise ValueError("runner's exact t-critical table supports at most 31 seeds")
        low = average - critical * standard_error
        high = average + critical * standard_error
    return {
        "mean": average,
        "se": standard_error,
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(len(data)),
    }


def exact_sign_test(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    positive = int(np.sum(data > 1e-12))
    negative = int(np.sum(data < -1e-12))
    ties = int(len(data) - positive - negative)
    non_ties = positive + negative
    if non_ties:
        tail = sum(math.comb(non_ties, value) for value in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / float(2**non_ties))
    else:
        p_value = 1.0
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "non_ties": non_ties,
        "p_value_two_sided": float(p_value),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_seed(*parts: int) -> int:
    return int(
        np.random.SeedSequence([int(value) for value in parts]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                for row in rows:
                    text.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing typed-audit config: {resolved}")
    config_hash = sha256_file(resolved)
    if config_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "typed-audit config is not the frozen runner version: "
            f"expected {EXPECTED_CONFIG_SHA256}, got {config_hash}"
        )
    config = json.loads(resolved.read_text(encoding="utf-8"))
    return config, config_hash


def protocol_path(config: Mapping[str, Any]) -> Path:
    return (REPO / str(config["protocol_file"])).resolve()


def base_checkpoint_path(config: Mapping[str, Any], seed: int) -> Path:
    pattern = str(config["base_checkpoint_pattern"])
    return (REPO / pattern.format(seed=int(seed))).resolve()


def training_environment_seed(namespace: int, policy_seed: int, episode: int) -> int:
    return int(namespace) + (int(policy_seed) % 100_000) * 1_000 + int(episode)


def runtime_config(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode == "formal":
        return {
            "mode": mode,
            "policy_seeds": [int(value) for value in config["policy_seeds"]],
            "response_episodes": int(config["response_episodes"]),
            "evaluation_seeds": [int(value) for value in config["evaluation_seeds"]],
            "response_environment_seed_namespace": int(
                config["response_environment_seed_namespace"]
            ),
            "training_action_namespace": int(config["training_action_namespace"]),
            "evaluation_action_namespace": int(config["evaluation_action_namespace"]),
            "workers": int(config["workers"]),
            "output_subdirectory": "formal",
        }
    if mode != "engineering_smoke":
        raise ValueError(f"unknown execution mode: {mode}")
    smoke = config["engineering_smoke"]
    return {
        "mode": mode,
        "policy_seeds": [int(config["policy_seeds"][0])],
        "response_episodes": SMOKE_RESPONSE_EPISODES,
        "evaluation_seeds": [int(smoke["evaluation_seed"])],
        "response_environment_seed_namespace": int(
            smoke["response_environment_seed_namespace"]
        ),
        "training_action_namespace": int(smoke["training_action_namespace"]),
        "evaluation_action_namespace": int(smoke["evaluation_action_namespace"]),
        "workers": 1,
        "output_subdirectory": "engineering_smoke",
    }


def validate_namespace_isolation(config: Mapping[str, Any]) -> dict[str, Any]:
    formal_training_paths = {
        training_environment_seed(
            int(config["response_environment_seed_namespace"]), seed, episode
        )
        for seed in config["policy_seeds"]
        for episode in range(int(config["response_episodes"]))
    }
    smoke = config["engineering_smoke"]
    smoke_seed = int(config["policy_seeds"][0])
    smoke_training_paths = {
        training_environment_seed(
            int(smoke["response_environment_seed_namespace"]), smoke_seed, episode
        )
        for episode in range(SMOKE_RESPONSE_EPISODES)
    }
    formal_evaluation = {int(value) for value in config["evaluation_seeds"]}
    smoke_evaluation = {int(smoke["evaluation_seed"])}
    formal_actions = {
        int(config["training_action_namespace"]),
        int(config["evaluation_action_namespace"]),
    }
    smoke_actions = {
        int(smoke["training_action_namespace"]),
        int(smoke["evaluation_action_namespace"]),
    }
    path_sets = {
        "formal_training": formal_training_paths,
        "formal_evaluation": formal_evaluation,
        "smoke_training": smoke_training_paths,
        "smoke_evaluation": smoke_evaluation,
    }
    names = list(path_sets)
    overlaps: dict[str, list[int]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(path_sets[left] & path_sets[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    if overlaps:
        raise RuntimeError(f"environment seed namespaces overlap: {overlaps}")
    if formal_actions & smoke_actions:
        raise RuntimeError("formal and smoke action namespaces overlap")
    if len(formal_actions) != 2 or len(smoke_actions) != 2:
        raise RuntimeError("training and evaluation action namespaces overlap")
    return {
        "formal_training_path_count": len(formal_training_paths),
        "formal_evaluation_path_count": len(formal_evaluation),
        "smoke_training_path_count": len(smoke_training_paths),
        "smoke_evaluation_path_count": len(smoke_evaluation),
        "environment_seed_sets_pairwise_disjoint": True,
        "formal_smoke_action_namespaces_disjoint": True,
        "within_mode_action_namespaces_disjoint": True,
    }


def validate_static_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if str(config.get("protocol_version")) != PROTOCOL_VERSION:
        raise ValueError("typed-audit protocol version mismatch")
    if str(config.get("freeze_status")) != "frozen_pre_outcome":
        raise RuntimeError("typed-audit config is not frozen pre-outcome")
    if not bool(config.get("formal_execution_allowed")):
        raise RuntimeError("formal execution is disabled by the frozen config")
    if tuple(config.get("methods", ())) != METHODS:
        raise ValueError(f"typed-audit method order must be {METHODS}")
    if tuple(int(value) for value in config["policy_seeds"]) != FORMAL_POLICY_SEEDS:
        raise ValueError("formal policy seed set changed")
    if tuple(int(value) for value in config["evaluation_seeds"]) != FORMAL_EVALUATION_SEEDS:
        raise ValueError("formal evaluation seed set changed")
    if int(config["response_episodes"]) != FORMAL_RESPONSE_EPISODES:
        raise ValueError("formal response-training budget changed")
    if int(config["episode_steps"]) != 100:
        raise ValueError("typed-audit episode horizon must remain 100")
    if not math.isclose(float(config["reward_scale"]), REWARD_SCALE, abs_tol=1e-15):
        raise ValueError("typed-audit financial reward scale must remain 100")
    if str(config["official_commit"]) != OFFICIAL_COMMIT:
        raise RuntimeError("official learner commit differs from the pinned adapter")
    if str(config["protocol_file_sha256"]) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash changed in the config")
    actual_protocol_hash = sha256_file(protocol_path(config))
    if actual_protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "typed-audit protocol file hash mismatch: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, got {actual_protocol_hash}"
        )
    spec = TypedAuditSpec(**config["audit"])
    if not math.isclose(spec.action_cost_coefficient, 0.20, abs_tol=1e-15):
        raise ValueError("action-cost coefficient changed")
    if not math.isclose(spec.actor_coefficient, 0.30, abs_tol=1e-15):
        raise ValueError("actor audit coefficient changed")
    if tuple(spec.acceptable_action_indices) != (0, 1, 2):
        raise ValueError("typed response set changed")
    if str(config.get("routing_allocation_mode")) != "soft_rank":
        raise ValueError("base environment allocation mode changed")
    return {
        "protocol_sha256": actual_protocol_hash,
        "namespace_isolation": validate_namespace_isolation(config),
    }


def load_base_checkpoint(
    config: Mapping[str, Any], seed: int, validate_hash: bool = True
) -> tuple[OfficialMAPPO, dict[str, Any]]:
    path = base_checkpoint_path(config, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing base checkpoint: {path}")
    actual_hash = sha256_file(path)
    expected_hash = str(config["base_checkpoint_sha256"][str(int(seed))])
    if validate_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"base checkpoint hash mismatch for seed {seed}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("completed_episodes", -1)) != 384:
        raise RuntimeError(f"base checkpoint seed {seed} is not complete")
    learner_payload = payload.get("learner")
    if not isinstance(learner_payload, Mapping):
        raise RuntimeError(f"base checkpoint seed {seed} has no learner payload")
    learner = OfficialMAPPO.from_payload(dict(learner_payload))
    if learner.source_commit != OFFICIAL_COMMIT:
        raise RuntimeError("loaded learner source commit mismatch")
    if learner.config.centralized_value:
        raise RuntimeError("typed-audit protocol requires decentralized IPPO values")
    if not math.isclose(learner.config.reward_scale, REWARD_SCALE, abs_tol=1e-15):
        raise RuntimeError("base learner reward scale is not 100")
    if learner.config.episode_steps != int(config["episode_steps"]):
        raise RuntimeError("base learner horizon mismatch")
    if learner.config.n_agents != int(config["market"]["n_agents"]):
        raise RuntimeError("base learner agent count mismatch")
    return learner, {
        "path": str(path.relative_to(REPO)),
        "sha256": actual_hash,
        "completed_episodes": int(payload["completed_episodes"]),
        "model_digest": learner.model_digest(),
    }


def validate_smoke_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    output = (REPO / str(config["output_dir"]) / "engineering_smoke").resolve()
    required = {
        name: output / name
        for name in (
            "manifest.json",
            "run_state.json",
            "failures.json",
            "rows.json",
            "summary.json",
        )
    }
    missing = [str(path.relative_to(REPO)) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"formal run requires completed engineering smoke: {missing}")
    manifest = json.loads(required["manifest.json"].read_text(encoding="utf-8"))
    state = json.loads(required["run_state.json"].read_text(encoding="utf-8"))
    failures = json.loads(required["failures.json"].read_text(encoding="utf-8"))
    rows = json.loads(required["rows.json"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary.json"].read_text(encoding="utf-8"))
    if manifest.get("mode") != "engineering_smoke":
        raise RuntimeError("engineering-smoke manifest has the wrong mode")
    if manifest.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("engineering-smoke config hash differs from formal runner")
    if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("engineering-smoke protocol hash differs from formal runner")
    if manifest.get("code_sha256") != code_hashes():
        raise RuntimeError("engineering smoke used different executable code")
    if state.get("status") != "complete" or state.get("failed_policy_seeds"):
        raise RuntimeError("engineering smoke did not complete without failed seeds")
    if failures != []:
        raise RuntimeError("engineering smoke retained one or more failures")
    if len(rows) != 1 or int(rows[0].get("policy_seed", -1)) != FORMAL_POLICY_SEEDS[0]:
        raise RuntimeError("engineering smoke has an incomplete seed denominator")
    integrity = summary.get("integrity", {})
    required_integrity = (
        "all_policy_seeds_complete",
        "zero_failures",
        "all_training_paths_equal",
        "all_training_action_uniforms_equal",
        "all_evaluation_paths_equal",
        "all_evaluation_action_uniforms_equal",
        "routing_disabled_every_step",
        "no_seed_excluded",
    )
    if not all(integrity.get(key) is True for key in required_integrity):
        raise RuntimeError("engineering smoke failed one or more integrity gates")
    for method in METHODS:
        for metric in ("q_v", "q_x"):
            value = rows[0].get(f"{method}__{metric}")
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"engineering smoke lacks the {method} {metric} denominator"
                )
    return {
        "output": str(output.relative_to(REPO)),
        "manifest_sha256": sha256_file(required["manifest.json"]),
        "run_state_sha256": sha256_file(required["run_state.json"]),
        "summary_sha256": sha256_file(required["summary.json"]),
        "zero_failures": True,
        "full_audit_denominators": True,
    }


def preflight(
    config: Mapping[str, Any], config_hash: str, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    static = validate_static_config(config)
    official = verify_official_source(str(config["official_source_root"]))
    docker_id = resolve_docker_image_id(str(config["docker_image"]))
    if docker_id != str(config["docker_image_id"]):
        raise RuntimeError(
            f"Docker image ID mismatch: expected {config['docker_image_id']}, got {docker_id}"
        )
    if runtime["mode"] == "formal":
        runner_path = Path(__file__).resolve()
        if not git_file_is_tracked(runner_path):
            raise RuntimeError("formal typed-audit runner must be committed")
        violations = formal_worktree_violations(str(config["output_dir"]))
        if violations:
            raise RuntimeError(
                "formal typed-audit run requires a clean committed worktree; "
                f"violations={violations}"
            )
    checkpoints = {}
    for seed in runtime["policy_seeds"]:
        learner, receipt = load_base_checkpoint(config, int(seed))
        checkpoints[str(seed)] = receipt
        del learner
    receipt = {
        "checked_at": utc_now(),
        "mode": runtime["mode"],
        "config_sha256": config_hash,
        "protocol_sha256": static["protocol_sha256"],
        "namespace_isolation": static["namespace_isolation"],
        "official_source": official,
        "official_commit": OFFICIAL_COMMIT,
        "docker_image": str(config["docker_image"]),
        "docker_image_id": docker_id,
        "git_commit": current_git_commit(),
        "git_worktree_clean_required": runtime["mode"] == "formal",
        "base_checkpoints": checkpoints,
    }
    if runtime["mode"] == "formal":
        receipt["engineering_smoke_gate"] = validate_smoke_gate(config)
    return receipt


def environment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    market = copy.deepcopy(dict(config["market"]))
    market["episode_steps"] = int(config["episode_steps"])
    market["routing_allocation_mode"] = str(config["routing_allocation_mode"])
    return market


def pairwise_agreement(actions: np.ndarray) -> float:
    values = np.asarray(actions, dtype=np.int64).reshape(-1)
    matches = 0
    pairs = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            pairs += 1
            matches += int(values[left] == values[right])
    return float(matches / pairs) if pairs else 0.0


def action_entropy(actions: np.ndarray, n_actions: int) -> tuple[float, float]:
    counts = np.bincount(
        np.asarray(actions, dtype=np.int64).reshape(-1), minlength=int(n_actions)
    ).astype(float)
    probabilities = counts[counts > 0.0] / float(np.sum(counts))
    raw = float(-np.sum(probabilities * np.log(probabilities)))
    normalized = raw / math.log(float(n_actions))
    return raw, normalized


def build_step_record(
    *,
    config: Mapping[str, Any],
    fingerprint: str,
    mode: str,
    phase: str,
    policy_seed: int,
    environment_seed: int,
    episode: int,
    method: str,
    step: int,
    action_namespace: int,
    model_digest: str,
    base_model_digest: str,
    actions: np.ndarray,
    uniforms: np.ndarray,
    audit: Mapping[str, np.ndarray],
    response_mass: np.ndarray,
    raw_rewards: np.ndarray,
    learning_rewards: np.ndarray,
    action_costs: np.ndarray,
    next_state: Mapping[str, Any],
) -> dict[str, Any]:
    info = next_state["info"]
    valid = np.asarray(audit["valid_mask"], dtype=bool)
    typed_response = np.asarray(audit["response_mask"], dtype=bool)
    response = np.asarray(audit["measurement_response_mask"], dtype=bool)
    if np.any(typed_response != (response & valid[..., None])):
        raise RuntimeError("typed response mask does not equal validity times fixed set")
    penalties = np.asarray(info["inventory_penalty_by_dealer"], dtype=float)
    inventories = np.asarray(info["inventory_by_dealer"], dtype=float)
    raw_float32 = np.asarray(raw_rewards, dtype=np.float32)
    raw = raw_float32.astype(float)
    learning = np.asarray(learning_rewards, dtype=float)
    costs_float32 = np.asarray(action_costs, dtype=np.float32)
    costs = costs_float32.astype(float)
    gross = raw + penalties
    half_spreads = np.asarray(actions, dtype=float).reshape(-1) + 1.0
    entropy, normalized_entropy = action_entropy(actions, response.shape[-1])
    audit_loss = np.where(
        valid,
        -np.log(np.maximum(response_mass, float(config["audit"]["probability_floor"]))),
        0.0,
    )
    invalid_contribution = float(np.sum(audit_loss[~valid]))
    if invalid_contribution != 0.0:
        raise RuntimeError("canonical invalid rows contributed to the audit loss")
    routing_tv = float(info["routing_total_variation"])
    targeted_quantity = float(info["realized_targeted_quantity"])
    if abs(routing_tv) > 1e-12 or abs(targeted_quantity) > 1e-12:
        raise RuntimeError(
            "routing-disabled typed-audit rollout changed allocation: "
            f"tv={routing_tv}, targeted_quantity={targeted_quantity}"
        )
    if bool(info.get("routing_enabled", True)):
        raise RuntimeError("typed-audit rollout did not disable routing")
    expected_learning = raw_float32 / REWARD_SCALE
    if method == "action_cost":
        expected_learning = expected_learning - np.float32(
            config["audit"]["action_cost_coefficient"]
        ) * costs_float32
    if not np.array_equal(
        np.asarray(learning_rewards, dtype=np.float32), expected_learning
    ):
        raise RuntimeError(f"learning reward formula mismatch for {method}")
    finite_arrays = (
        response_mass,
        raw,
        learning,
        costs,
        penalties,
        inventories,
        gross,
        half_spreads,
    )
    if not all(np.all(np.isfinite(values)) for values in finite_arrays):
        raise FloatingPointError("non-finite typed-audit step endpoint")
    return {
        "provenance": {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_fingerprint": fingerprint,
            "mode": mode,
            "phase": phase,
            "policy_seed": int(policy_seed),
            "environment_seed": int(environment_seed),
            "episode": int(episode),
            "method": method,
            "step": int(step),
            "action_namespace": int(action_namespace),
            "path_digest": str(info["path_digest"]),
            "action_uniform_digest": canonical_digest(uniforms.tolist()),
            "model_digest": model_digest,
            "base_model_digest": base_model_digest,
            "routing_enabled": False,
            "actor_signal": 0.0,
        },
        "audit": {
            "canonical_full_valid_mask": valid.tolist(),
            "typed_response_mask": typed_response.tolist(),
            "measurement_response_mask": response.tolist(),
            "shadow_response_probability": response_mass.tolist(),
            "valid_count": int(np.count_nonzero(valid)),
            "invalid_count": int(valid.size - np.count_nonzero(valid)),
            "invalid_audit_contribution": invalid_contribution,
        },
        "actions": np.asarray(actions, dtype=int).reshape(-1).tolist(),
        "action_uniforms": np.asarray(uniforms, dtype=float).tolist(),
        "financial_reward_raw": raw.tolist(),
        "inventory_penalty": penalties.tolist(),
        "gross_pnl_before_inventory_penalty": gross.tolist(),
        "learning_reward_scaled": learning.tolist(),
        "action_cost": costs.tolist(),
        "market": {
            "mean_financial_reward_raw": float(np.mean(raw)),
            "total_financial_reward_raw": float(np.sum(raw)),
            "mean_inventory_penalty": float(np.mean(penalties)),
            "total_inventory_penalty": float(np.sum(penalties)),
            "mean_gross_pnl": float(np.mean(gross)),
            "total_gross_pnl": float(np.sum(gross)),
            "inventories": inventories.tolist(),
            "mean_abs_inventory": float(np.mean(np.abs(inventories))),
            "cross_dealer_inventory_variance": float(np.var(inventories)),
            "mean_quoted_half_spread": float(np.mean(half_spreads)),
            "effective_half_spread": float(info["mean_execution_half_spread"]),
            "fill_ratio": float(info["fill_ratio"]),
            "volume": float(info["realized_flow_quantity"]),
            "customer_cost": float(info["customer_execution_cost_tick_units"]),
            "quote_dispersion": float(np.std(half_spreads)),
            "best_quote_tie_share": float(
                np.count_nonzero(half_spreads == np.min(half_spreads))
                / len(half_spreads)
            ),
            "pairwise_action_agreement": pairwise_agreement(actions),
            "action_entropy": entropy,
            "normalized_action_entropy": normalized_entropy,
            "routing_total_variation": routing_tv,
            "targeted_quantity": targeted_quantity,
        },
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("typed-audit rollout has no records")
    valid_values: list[float] = []
    invalid_values: list[float] = []
    valid_rows = 0
    total_rows = 0
    raw_rewards = None
    penalties = None
    gross = None
    for row in records:
        audit = row["audit"]
        valid = np.asarray(audit["canonical_full_valid_mask"], dtype=bool)
        q = np.asarray(audit["shadow_response_probability"], dtype=float)
        valid_values.extend(q[valid].tolist())
        invalid_values.extend(q[~valid].tolist())
        valid_rows += int(np.count_nonzero(valid))
        total_rows += int(valid.size)
        row_raw = np.asarray(row["financial_reward_raw"], dtype=float)
        row_penalty = np.asarray(row["inventory_penalty"], dtype=float)
        row_gross = np.asarray(row["gross_pnl_before_inventory_penalty"], dtype=float)
        raw_rewards = row_raw if raw_rewards is None else raw_rewards + row_raw
        penalties = row_penalty if penalties is None else penalties + row_penalty
        gross = row_gross if gross is None else gross + row_gross
    assert raw_rewards is not None and penalties is not None and gross is not None
    q_v = float(mean(valid_values)) if valid_values else None
    q_x = float(mean(invalid_values)) if invalid_values else None
    market_rows = [row["market"] for row in records]
    total_volume = float(sum(float(row["volume"]) for row in market_rows))
    total_customer_cost = float(sum(float(row["customer_cost"]) for row in market_rows))
    result = {
        "q_v": q_v,
        "q_x": q_x,
        "selectivity": (float(q_v - q_x) if q_v is not None and q_x is not None else None),
        "valid_audit_rate": float(valid_rows / total_rows),
        "valid_audit_rows": int(valid_rows),
        "invalid_audit_rows": int(total_rows - valid_rows),
        "total_audit_rows": int(total_rows),
        "invalid_audit_contribution": float(
            sum(float(row["audit"]["invalid_audit_contribution"]) for row in records)
        ),
        "aggregate_financial_reward": float(np.sum(raw_rewards)),
        "mean_per_dealer_financial_reward": float(np.mean(raw_rewards)),
        "aggregate_gross_pnl": float(np.sum(gross)),
        "mean_per_dealer_gross_pnl": float(np.mean(gross)),
        "aggregate_inventory_penalty": float(np.sum(penalties)),
        "mean_per_dealer_inventory_penalty": float(np.mean(penalties)),
        "mean_abs_inventory": float(mean(float(row["mean_abs_inventory"]) for row in market_rows)),
        "mean_cross_dealer_inventory_variance": float(
            mean(float(row["cross_dealer_inventory_variance"]) for row in market_rows)
        ),
        "mean_quoted_half_spread": float(
            mean(float(row["mean_quoted_half_spread"]) for row in market_rows)
        ),
        "effective_half_spread": (
            float(total_customer_cost / total_volume) if total_volume > 0.0 else 0.0
        ),
        "mean_fill_ratio": float(mean(float(row["fill_ratio"]) for row in market_rows)),
        "total_volume": total_volume,
        "total_customer_cost": total_customer_cost,
        "mean_quote_dispersion": float(
            mean(float(row["quote_dispersion"]) for row in market_rows)
        ),
        "mean_best_quote_tie_share": float(
            mean(float(row["best_quote_tie_share"]) for row in market_rows)
        ),
        "mean_pairwise_action_agreement": float(
            mean(float(row["pairwise_action_agreement"]) for row in market_rows)
        ),
        "mean_action_entropy": float(
            mean(float(row["action_entropy"]) for row in market_rows)
        ),
        "mean_normalized_action_entropy": float(
            mean(float(row["normalized_action_entropy"]) for row in market_rows)
        ),
        "routing_total_variation": float(
            sum(float(row["routing_total_variation"]) for row in market_rows)
        ),
        "targeted_quantity": float(sum(float(row["targeted_quantity"]) for row in market_rows)),
        "path_digest": str(records[0]["provenance"]["path_digest"]),
        "action_uniform_sequence_digest": digest(
            [row["provenance"]["action_uniform_digest"] for row in records]
        ),
        "finite": True,
    }
    numeric = [
        value
        for value in result.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise FloatingPointError("non-finite typed-audit path summary")
    if result["invalid_audit_contribution"] != 0.0:
        raise RuntimeError("invalid audit contribution is nonzero")
    if result["routing_total_variation"] != 0.0 or result["targeted_quantity"] != 0.0:
        raise RuntimeError("routing-disabled path has nonzero routing endpoints")
    return result


def rollout_arm(
    *,
    env: ABIDESDockerDealerEnv,
    learner: OfficialMAPPO,
    config: Mapping[str, Any],
    spec: TypedAuditSpec,
    fingerprint: str,
    runtime: Mapping[str, Any],
    phase: str,
    policy_seed: int,
    environment_seed: int,
    episode: int,
    method: str,
    action_namespace: int,
    train: bool,
    update_seed: int,
    base_model_digest: str,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown typed-audit method: {method}")
    state = env.reset(seed=int(environment_seed))
    rnn_actor, rnn_critic, masks = learner.initial_recurrent_states()
    buffer = None
    records: list[dict[str, Any]] = []
    rollout_model_digest = learner.model_digest()
    episode_failed = False
    try:
        for step in range(int(config["episode_steps"])):
            obs = actor_observations(state, 0.0)
            if not np.all(obs[:, -1] == 0.0):
                raise RuntimeError("typed-audit actor signal is not identically zero")
            critic_obs = obs.copy()
            if train:
                if buffer is None:
                    buffer = learner.new_buffer(obs, critic_obs)
                else:
                    buffer.obs[step, 0] = obs
                    buffer.share_obs[step, 0] = critic_obs
            canonical_audit = build_audit_batch(obs, spec, always_on=False)
            probabilities = learner.action_probabilities(
                canonical_audit["shadow_obs"], rnn_actor, masks
            )
            response_mass = np.sum(
                probabilities
                * np.asarray(
                    canonical_audit["measurement_response_mask"], dtype=np.float32
                ),
                axis=-1,
            )
            uniforms = np.random.default_rng(
                np.random.SeedSequence(
                    [int(environment_seed), int(step), int(action_namespace)]
                )
            ).random(learner.config.n_agents)
            actions, logp, values, rnn_actor_next, rnn_critic_next = learner.act(
                obs,
                critic_obs,
                rnn_actor,
                rnn_critic,
                masks,
                uniforms,
            )
            action_vector = actions[:, 0].astype(np.int64)
            measurement_response_mask = np.asarray(
                canonical_audit["measurement_response_mask"], dtype=bool
            )
            chosen_in_response = measurement_response_mask[
                np.arange(learner.config.n_agents), action_vector
            ]
            action_costs = (
                np.asarray(canonical_audit["valid_mask"], dtype=bool)
                & ~chosen_in_response
            ).astype(np.float32)
            next_state = env.step(
                action_vector,
                gate_active=False,
                routing_enabled=False,
                routing_strength=0.0,
            )
            raw_rewards = np.asarray(next_state["rewards"], dtype=np.float32)
            financial_scaled = raw_rewards / REWARD_SCALE
            if method == "action_cost":
                learning_rewards = financial_scaled - np.float32(
                    spec.action_cost_coefficient
                ) * action_costs
            else:
                learning_rewards = financial_scaled.copy()
            next_masks = (
                np.zeros_like(masks)
                if bool(next_state["done"])
                else np.ones_like(masks)
            )
            placeholder_obs = actor_observations(next_state, 0.0)
            placeholder_critic = placeholder_obs.copy()
            if train:
                assert buffer is not None
                learner.insert(
                    buffer,
                    placeholder_obs,
                    placeholder_critic,
                    rnn_actor_next,
                    rnn_critic_next,
                    actions,
                    logp,
                    values,
                    learning_rewards,
                    next_masks,
                )
            records.append(
                build_step_record(
                    config=config,
                    fingerprint=fingerprint,
                    mode=str(runtime["mode"]),
                    phase=phase,
                    policy_seed=policy_seed,
                    environment_seed=environment_seed,
                    episode=episode,
                    method=method,
                    step=step,
                    action_namespace=action_namespace,
                    model_digest=rollout_model_digest,
                    base_model_digest=base_model_digest,
                    actions=action_vector,
                    uniforms=uniforms,
                    audit=canonical_audit,
                    response_mass=response_mass,
                    raw_rewards=raw_rewards,
                    learning_rewards=learning_rewards,
                    action_costs=action_costs,
                    next_state=next_state,
                )
            )
            state = next_state
            rnn_actor = rnn_actor_next
            rnn_critic = rnn_critic_next
            masks = next_masks
        if not bool(state["done"]):
            raise RuntimeError("typed-audit episode did not reach its frozen horizon")
    except BaseException:
        episode_failed = True
        raise
    finally:
        if episode_failed:
            try:
                env.close_episode()
            except Exception:
                pass
        else:
            env.close_episode()
    update: dict[str, float] = {}
    if train:
        if buffer is None:
            raise RuntimeError("typed-audit training rollout produced no replay buffer")
        if method in {"always_on_actor", "full_ace"}:
            update = official_update_with_set_audit(
                learner,
                buffer,
                int(update_seed),
                spec,
                always_on=method == "always_on_actor",
            )
        else:
            update = learner.update(buffer, int(update_seed))
    return {
        "records": records,
        "summary": summarize_records(records),
        "update": update,
        "pre_update_model_digest": rollout_model_digest,
        "post_update_model_digest": learner.model_digest(),
    }


def common_rollout_integrity(rollouts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(rollouts) != METHODS:
        raise RuntimeError("typed-audit rollout method order changed")
    reference = rollouts[METHODS[0]]["records"]
    reference_paths = [row["provenance"]["path_digest"] for row in reference]
    reference_uniforms = [
        row["provenance"]["action_uniform_digest"] for row in reference
    ]
    path_equal = True
    uniforms_equal = True
    for method in METHODS[1:]:
        candidate = rollouts[method]["records"]
        path_equal &= [row["provenance"]["path_digest"] for row in candidate] == reference_paths
        uniforms_equal &= [
            row["provenance"]["action_uniform_digest"] for row in candidate
        ] == reference_uniforms
    if not path_equal:
        raise RuntimeError("typed-audit arms did not receive the same market path")
    if not uniforms_equal:
        raise RuntimeError("typed-audit arms did not receive the same action uniforms")
    return {
        "paths_equal": True,
        "action_uniforms_equal": True,
        "path_sequence_digest": digest(reference_paths),
        "action_uniform_sequence_digest": digest(reference_uniforms),
    }


def task_directory(output: Path, policy_seed: int) -> Path:
    return output / "seeds" / f"seed_{int(policy_seed)}"


def seed_result_path(output: Path, policy_seed: int) -> Path:
    return task_directory(output, policy_seed) / "result.json"


def count_gzip_jsonl_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_seed_result(
    result: Mapping[str, Any],
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected_episodes = int(runtime["response_episodes"])
    if int(result.get("completed_response_episodes", -1)) != expected_episodes:
        raise RuntimeError("seed result has an incomplete response-training budget")
    if len(result.get("training_history", ())) != expected_episodes:
        raise RuntimeError("seed result has incomplete training history")
    if len(result.get("training_integrity", ())) != expected_episodes:
        raise RuntimeError("seed result has incomplete training integrity rows")
    expected_paths = [int(value) for value in runtime["evaluation_seeds"]]
    if len(expected_paths) != len(set(expected_paths)):
        raise RuntimeError("runtime contains duplicate evaluation paths")
    observed_paths = [int(value) for value in result.get("evaluation_seeds", ())]
    if observed_paths != expected_paths:
        raise RuntimeError("seed result evaluation paths differ from the frozen runtime")
    evaluation = result.get("evaluation", {})
    per_path = list(evaluation.get("per_path", ()))
    expected_pairs = {(seed, method) for seed in expected_paths for method in METHODS}
    observed_pairs = [
        (int(row["evaluation_seed"]), str(row["method"])) for row in per_path
    ]
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        raise RuntimeError("seed result lacks one exact method-by-evaluation-path row")
    total_valid = {method: 0 for method in METHODS}
    total_invalid = {method: 0 for method in METHODS}
    expected_audit_rows = int(config["episode_steps"]) * int(config["market"]["n_agents"])
    for row in per_path:
        method = str(row["method"])
        valid = int(row["valid_audit_rows"])
        invalid = int(row["invalid_audit_rows"])
        total = int(row["total_audit_rows"])
        if total != expected_audit_rows or valid + invalid != total:
            raise RuntimeError("evaluation path has an incomplete audit denominator")
        if valid and (row.get("q_v") is None or not math.isfinite(float(row["q_v"]))):
            raise RuntimeError("evaluation path has invalid q_v")
        if invalid and (row.get("q_x") is None or not math.isfinite(float(row["q_x"]))):
            raise RuntimeError("evaluation path has invalid q_x")
        total_valid[method] += valid
        total_invalid[method] += invalid
    if any(total_valid[method] <= 0 or total_invalid[method] <= 0 for method in METHODS):
        raise RuntimeError("seed result does not retain both audit denominators")
    expected_steps = len(expected_paths) * len(METHODS) * int(config["episode_steps"])
    if int(evaluation.get("complete_step_record_count", -1)) != expected_steps:
        raise RuntimeError("seed result has an incomplete evaluation step count")
    if len(evaluation.get("integrity", ())) != len(expected_paths):
        raise RuntimeError("seed result has incomplete evaluation integrity rows")
    step_path = REPO / str(result["evaluation_step_records_file"])
    if not step_path.is_file() or sha256_file(step_path) != result.get(
        "evaluation_step_records_sha256"
    ):
        raise RuntimeError("seed result has an invalid evaluation step artifact")
    if count_gzip_jsonl_rows(step_path) != expected_steps:
        raise RuntimeError("evaluation step artifact has an incomplete row denominator")


def run_policy_seed(
    config: dict[str, Any],
    runtime: dict[str, Any],
    output_raw: str,
    fingerprint: str,
    policy_seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = Path(output_raw)
    seed_dir = task_directory(output, policy_seed)
    result_path = seed_result_path(output, policy_seed)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"seed result fingerprint mismatch: {result_path}")
        validate_seed_result(result, runtime, config)
        return result
    base, base_receipt = load_base_checkpoint(config, policy_seed)
    base_digest = base.model_digest()
    learners = {method: OfficialMAPPO.from_payload(base.payload()) for method in METHODS}
    initial_digests = {method: learner.model_digest() for method, learner in learners.items()}
    if set(initial_digests.values()) != {base_digest}:
        raise RuntimeError("typed-audit arms did not clone the same base model")
    spec = TypedAuditSpec(**config["audit"])
    checkpoint = seed_dir / "training_state.pt"
    completed = 0
    training_history: list[dict[str, Any]] = []
    training_integrity: list[dict[str, Any]] = []
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(f"training checkpoint fingerprint mismatch: {checkpoint}")
        if int(payload.get("policy_seed", -1)) != int(policy_seed):
            raise RuntimeError(f"training checkpoint seed mismatch: {checkpoint}")
        if payload.get("base_model_digest") != base_digest:
            raise RuntimeError(f"training checkpoint base digest mismatch: {checkpoint}")
        completed = int(payload["completed_response_episodes"])
        training_history = list(payload["training_history"])
        training_integrity = list(payload["training_integrity"])
        if not 0 <= completed <= int(runtime["response_episodes"]):
            raise RuntimeError("training checkpoint completion is outside the runtime budget")
        if len(training_history) != completed or len(training_integrity) != completed:
            raise RuntimeError("training checkpoint history does not match completion")
        if set(payload.get("learners", {})) != set(METHODS):
            raise RuntimeError("training checkpoint does not contain all frozen arms")
        learners = {
            method: OfficialMAPPO.from_payload(payload["learners"][method])
            for method in METHODS
        }
    docker = ABIDESDockerConfig(image=str(config["docker_image"]))
    with ABIDESDockerDealerEnv(environment_config(config), docker) as env:
        while completed < int(runtime["response_episodes"]):
            episode = completed
            environment_seed = training_environment_seed(
                int(runtime["response_environment_seed_namespace"]),
                policy_seed,
                episode,
            )
            update_seed = deterministic_seed(
                policy_seed,
                episode,
                int(runtime["training_action_namespace"]),
                17,
            )
            rollouts: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                rollouts[method] = rollout_arm(
                    env=env,
                    learner=learners[method],
                    config=config,
                    spec=spec,
                    fingerprint=fingerprint,
                    runtime=runtime,
                    phase="response_training",
                    policy_seed=policy_seed,
                    environment_seed=environment_seed,
                    episode=episode,
                    method=method,
                    action_namespace=int(runtime["training_action_namespace"]),
                    train=True,
                    update_seed=update_seed,
                    base_model_digest=base_digest,
                )
            integrity = common_rollout_integrity(rollouts)
            completed += 1
            training_integrity.append(
                {
                    "completed_response_episodes": completed,
                    "environment_seed": environment_seed,
                    "update_seed": update_seed,
                    **integrity,
                }
            )
            training_history.append(
                {
                    "completed_response_episodes": completed,
                    "environment_seed": environment_seed,
                    "methods": {
                        method: {
                            "path_summary": rollouts[method]["summary"],
                            "update": rollouts[method]["update"],
                            "pre_update_model_digest": rollouts[method][
                                "pre_update_model_digest"
                            ],
                            "post_update_model_digest": rollouts[method][
                                "post_update_model_digest"
                            ],
                        }
                        for method in METHODS
                    },
                }
            )
            if (
                completed % int(config["checkpoint_interval"]) == 0
                or completed == int(runtime["response_episodes"])
            ):
                atomic_torch_save(
                    checkpoint,
                    {
                        "protocol_fingerprint": fingerprint,
                        "policy_seed": int(policy_seed),
                        "base_checkpoint_sha256": base_receipt["sha256"],
                        "base_model_digest": base_digest,
                        "initial_model_digests": initial_digests,
                        "completed_response_episodes": completed,
                        "learners": {
                            method: learner.payload() for method, learner in learners.items()
                        },
                        "training_history": training_history,
                        "training_integrity": training_integrity,
                        "saved_at": utc_now(),
                    },
                )
                append_logbook(
                    seed_dir / "progress.jsonl",
                    {
                        "event": "response_checkpoint",
                        "policy_seed": policy_seed,
                        "completed_response_episodes": completed,
                        "target_response_episodes": int(runtime["response_episodes"]),
                        "checkpoint_sha256": sha256_file(checkpoint),
                    },
                )
        evaluation_rows: list[dict[str, Any]] = []
        per_path: list[dict[str, Any]] = []
        evaluation_integrity: list[dict[str, Any]] = []
        for path_index, evaluation_seed in enumerate(runtime["evaluation_seeds"]):
            rollouts = {}
            for method in METHODS:
                rollouts[method] = rollout_arm(
                    env=env,
                    learner=learners[method],
                    config=config,
                    spec=spec,
                    fingerprint=fingerprint,
                    runtime=runtime,
                    phase="frozen_policy_evaluation",
                    policy_seed=policy_seed,
                    environment_seed=int(evaluation_seed),
                    episode=path_index,
                    method=method,
                    action_namespace=int(runtime["evaluation_action_namespace"]),
                    train=False,
                    update_seed=0,
                    base_model_digest=base_digest,
                )
            evaluation_integrity.append(
                {
                    "evaluation_seed": int(evaluation_seed),
                    **common_rollout_integrity(rollouts),
                }
            )
            for method in METHODS:
                evaluation_rows.extend(rollouts[method]["records"])
                per_path.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "protocol_fingerprint": fingerprint,
                        "mode": runtime["mode"],
                        "policy_seed": int(policy_seed),
                        "evaluation_seed": int(evaluation_seed),
                        "method": method,
                        "model_digest": learners[method].model_digest(),
                        "base_model_digest": base_digest,
                        **rollouts[method]["summary"],
                    }
                )
    step_path = seed_dir / "evaluation_step_records.jsonl.gz"
    write_jsonl_gz(step_path, evaluation_rows)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "mode": runtime["mode"],
        "policy_seed": int(policy_seed),
        "base_checkpoint": base_receipt,
        "base_model_digest": base_digest,
        "initial_model_digests": initial_digests,
        "final_model_digests": {
            method: learner.model_digest() for method, learner in learners.items()
        },
        "completed_response_episodes": completed,
        "evaluation_seeds": [int(value) for value in runtime["evaluation_seeds"]],
        "training_history": training_history,
        "training_integrity": training_integrity,
        "evaluation": {
            "per_path": per_path,
            "integrity": evaluation_integrity,
            "complete_step_record_count": len(evaluation_rows),
        },
        "evaluation_step_records_file": str(step_path.relative_to(REPO)),
        "evaluation_step_records_sha256": sha256_file(step_path),
        "elapsed_seconds": float(time.perf_counter() - started),
        "completed_at": utc_now(),
    }
    validate_seed_result(result, runtime, config)
    atomic_write_json(result_path, result)
    append_logbook(
        seed_dir / "progress.jsonl",
        {
            "event": "seed_complete",
            "policy_seed": policy_seed,
            "result_sha256": sha256_file(result_path),
        },
    )
    return result


def mean_optional(values: Sequence[Any]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return float(mean(retained)) if retained else None


def aggregate_seed_rows(
    results: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> list[dict[str, Any]]:
    rows = []
    observed_seed_ids = [int(result["policy_seed"]) for result in results]
    if len(observed_seed_ids) != len(set(observed_seed_ids)):
        raise RuntimeError("duplicate policy seed result detected")
    for result in sorted(results, key=lambda item: int(item["policy_seed"])):
        per_path = list(result["evaluation"]["per_path"])
        seed_row: dict[str, Any] = {"policy_seed": int(result["policy_seed"])}
        for method in methods:
            method_rows = [row for row in per_path if row["method"] == method]
            if len(method_rows) == 0:
                raise RuntimeError(
                    f"seed {result['policy_seed']} has no evaluation rows for {method}"
                )
            expected_paths = {int(value) for value in result.get("evaluation_seeds", [])}
            observed_paths = {int(row["evaluation_seed"]) for row in method_rows}
            if (
                not expected_paths
                or observed_paths != expected_paths
                or len(method_rows) != len(expected_paths)
            ):
                raise RuntimeError("seed result has incomplete evaluation paths")
            valid_rows = int(sum(int(row["valid_audit_rows"]) for row in method_rows))
            invalid_rows = int(sum(int(row["invalid_audit_rows"]) for row in method_rows))
            total_rows = int(sum(int(row["total_audit_rows"]) for row in method_rows))
            if total_rows <= 0 or valid_rows <= 0 or invalid_rows <= 0:
                raise RuntimeError("seed summary lacks a complete audit denominator")
            for row in method_rows:
                if int(row["valid_audit_rows"]) > 0 and row.get("q_v") is None:
                    raise RuntimeError("evaluation row silently dropped a q_v denominator")
                if int(row["invalid_audit_rows"]) > 0 and row.get("q_x") is None:
                    raise RuntimeError("evaluation row silently dropped a q_x denominator")
            q_v_numerator = sum(
                float(row["q_v"]) * int(row["valid_audit_rows"])
                for row in method_rows
                if row["q_v"] is not None
            )
            q_x_numerator = sum(
                float(row["q_x"]) * int(row["invalid_audit_rows"])
                for row in method_rows
                if row["q_x"] is not None
            )
            seed_q_v = float(q_v_numerator / valid_rows)
            seed_q_x = float(q_x_numerator / invalid_rows)
            seed_row[f"{method}__q_v"] = seed_q_v
            seed_row[f"{method}__q_x"] = seed_q_x
            seed_row[f"{method}__selectivity"] = (
                float(seed_q_v - seed_q_x)
                if seed_q_v is not None and seed_q_x is not None
                else None
            )
            seed_row[f"{method}__valid_audit_rate"] = float(valid_rows / total_rows)
            for metric in PRIMARY_METRICS:
                if metric in {"q_v", "q_x", "selectivity", "valid_audit_rate"}:
                    continue
                seed_row[f"{method}__{metric}"] = mean_optional(
                    [row[metric] for row in method_rows]
                )
            seed_row[f"{method}__valid_audit_rows"] = valid_rows
            seed_row[f"{method}__invalid_audit_rows"] = invalid_rows
            seed_row[f"{method}__total_audit_rows"] = total_rows
        rows.append(seed_row)
    return rows


def aggregate_summary(
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_rows = aggregate_seed_rows(results, METHODS)
    method_summary: dict[str, Any] = {}
    for method in METHODS:
        method_summary[method] = {}
        for metric in PRIMARY_METRICS:
            values = [row[f"{method}__{metric}"] for row in seed_rows]
            retained = [float(value) for value in values if value is not None]
            method_summary[method][metric] = (
                t_interval(retained) if retained else None
            )
    contrasts: dict[str, Any] = {}
    for label, left, right in (
        ("full_ace_minus_vanilla_ippo", "full_ace", "vanilla_ippo"),
        ("full_ace_minus_always_on_actor", "full_ace", "always_on_actor"),
    ):
        contrasts[label] = {}
        for metric in PRIMARY_METRICS:
            paired = []
            for row in seed_rows:
                left_value = row[f"{left}__{metric}"]
                right_value = row[f"{right}__{metric}"]
                if left_value is not None and right_value is not None:
                    paired.append(float(left_value) - float(right_value))
            contrasts[label][metric] = (
                {"t_interval": t_interval(paired), "sign_test": exact_sign_test(paired)}
                if paired
                else None
            )
    expected_seeds = {int(value) for value in runtime["policy_seeds"]}
    completed_seeds = {int(result["policy_seed"]) for result in results}
    integrity = {
        "expected_policy_seeds": sorted(expected_seeds),
        "completed_policy_seeds": sorted(completed_seeds),
        "all_policy_seeds_complete": completed_seeds == expected_seeds,
        "failure_count": len(failures),
        "zero_failures": len(failures) == 0,
        "all_training_paths_equal": bool(results) and all(
            all(bool(row["paths_equal"]) for row in result["training_integrity"])
            for result in results
        ),
        "all_training_action_uniforms_equal": bool(results) and all(
            all(bool(row["action_uniforms_equal"]) for row in result["training_integrity"])
            for result in results
        ),
        "all_evaluation_paths_equal": bool(results) and all(
            all(bool(row["paths_equal"]) for row in result["evaluation"]["integrity"])
            for result in results
        ),
        "all_evaluation_action_uniforms_equal": bool(results) and all(
            all(bool(row["action_uniforms_equal"]) for row in result["evaluation"]["integrity"])
            for result in results
        ),
        "routing_disabled_every_step": bool(results) and all(
            float(path["routing_total_variation"]) == 0.0
            and float(path["targeted_quantity"]) == 0.0
            for result in results
            for path in result["evaluation"]["per_path"]
        ),
        "no_seed_excluded": completed_seeds == expected_seeds and len(failures) == 0,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": runtime["mode"],
        "claim_boundary": (
            "engineering evidence only; never pooled with formal results"
            if runtime["mode"] == "engineering_smoke"
            else "exploratory evidence for the frozen tested market, rule, horizon, and seeds"
        ),
        "aggregation": "evaluation paths averaged within policy seed; paired t interval across policy seeds",
        "method_summary": method_summary,
        "paired_contrasts": contrasts,
        "integrity": integrity,
        "generated_at": utc_now(),
    }
    return seed_rows, summary


def execute(config_path: Path, mode: str) -> int:
    config, config_hash = load_config(config_path)
    runtime = runtime_config(config, mode)
    executable_hashes = code_hashes()
    runtime_fingerprint = digest(
        {
            "protocol_version": PROTOCOL_VERSION,
            "config_sha256": config_hash,
            "protocol_sha256": str(config["protocol_file_sha256"]),
            "code_sha256": executable_hashes,
            "runtime": runtime,
        }
    )
    output = (REPO / str(config["output_dir"]) / runtime["output_subdirectory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with RunLock(output / ".run.lock", runtime_fingerprint):
        receipt = preflight(config, config_hash, runtime)
        manifest_path = output / "manifest.json"
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_fingerprint": runtime_fingerprint,
            "mode": mode,
            "config_file": str(config_path.expanduser().resolve().relative_to(REPO)),
            "config_sha256": config_hash,
            "protocol_file": str(protocol_path(config).relative_to(REPO)),
            "protocol_sha256": str(config["protocol_file_sha256"]),
            "runner_file": str(Path(__file__).resolve().relative_to(REPO)),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "code_sha256": executable_hashes,
            "command": [sys.executable, *sys.argv],
            "runtime": runtime,
            "preflight": receipt,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "created_at": utc_now(),
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("protocol_fingerprint") != runtime_fingerprint:
                raise RuntimeError("existing output manifest belongs to another protocol")
            if previous.get("runner_sha256") != manifest["runner_sha256"]:
                raise RuntimeError("runner changed after this output namespace was created")
        else:
            atomic_write_json(manifest_path, manifest)
        state_path = output / "run_state.json"
        failures_path = output / "failures.json"
        if state_path.is_file():
            previous_state = json.loads(state_path.read_text(encoding="utf-8"))
            previous_failures = (
                json.loads(failures_path.read_text(encoding="utf-8"))
                if failures_path.is_file()
                else []
            )
            if previous_state.get("status") == "complete":
                for name in ("rows", "summary", "failures"):
                    artifact = output / f"{name}.json"
                    expected = previous_state.get(f"{name}_sha256")
                    if not artifact.is_file() or sha256_file(artifact) != expected:
                        raise RuntimeError(f"completed run has an invalid {name} artifact")
                return 0
            if previous_state.get("status") == "failed" or previous_failures:
                raise RuntimeError(
                    "failed typed-audit namespace is retained and cannot be retried"
                )
        failures: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        seeds = [int(value) for value in runtime["policy_seeds"]]
        atomic_write_json(failures_path, failures)
        atomic_write_json(
            state_path,
            {
                "status": "running",
                "mode": mode,
                "protocol_fingerprint": runtime_fingerprint,
                "expected_policy_seeds": seeds,
                "completed_policy_seeds": [],
                "failed_policy_seeds": [],
                "updated_at": utc_now(),
            },
        )
        with ProcessPoolExecutor(max_workers=int(runtime["workers"])) as executor:
            futures = {
                executor.submit(
                    run_policy_seed,
                    dict(config),
                    dict(runtime),
                    str(output),
                    runtime_fingerprint,
                    seed,
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    append_logbook(
                        output / "run_log.jsonl",
                        {"event": "policy_seed_complete", "policy_seed": seed},
                    )
                except Exception as error:  # retain every failure; no replacement
                    failure = {
                        "policy_seed": seed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "failed_at": utc_now(),
                    }
                    failures.append(failure)
                    append_logbook(
                        output / "run_log.jsonl",
                        {"event": "policy_seed_failed", **failure},
                    )
                atomic_write_json(failures_path, failures)
                atomic_write_json(
                    state_path,
                    {
                        "status": "running",
                        "mode": mode,
                        "protocol_fingerprint": runtime_fingerprint,
                        "expected_policy_seeds": seeds,
                        "completed_policy_seeds": sorted(
                            int(item["policy_seed"]) for item in results
                        ),
                        "failed_policy_seeds": sorted(
                            int(item["policy_seed"]) for item in failures
                        ),
                        "updated_at": utc_now(),
                    },
                )
        rows, summary = aggregate_summary(config, runtime, results, failures)
        atomic_write_json(output / "rows.json", rows)
        atomic_write_json(output / "summary.json", summary)
        complete = bool(summary["integrity"]["no_seed_excluded"])
        atomic_write_json(
            state_path,
            {
                "status": "complete" if complete else "failed",
                "mode": mode,
                "protocol_fingerprint": runtime_fingerprint,
                "expected_policy_seeds": seeds,
                "completed_policy_seeds": sorted(
                    int(item["policy_seed"]) for item in results
                ),
                "failed_policy_seeds": sorted(
                    int(item["policy_seed"]) for item in failures
                ),
                "rows_sha256": sha256_file(output / "rows.json"),
                "summary_sha256": sha256_file(output / "summary.json"),
                "failures_sha256": sha256_file(output / "failures.json"),
                "updated_at": utc_now(),
            },
        )
    return 0 if complete else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Frozen abides_typed_audit_ippo_v2 JSON configuration.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--engineering-smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "engineering_smoke" if args.engineering_smoke else "formal"
    return execute(args.config, mode)


if __name__ == "__main__":
    raise SystemExit(main())
