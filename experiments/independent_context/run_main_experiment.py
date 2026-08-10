#!/usr/bin/env python3
"""Frozen E-main multi-context, set-valued audit mechanism assay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


EXPECTED_PROTOCOL_SHA256 = (
    "18611fef078664e7633e24f6a436c08df12492df5ea9f090532f200eade3f5ed"
)
CONTEXTS = ("V", "I", "P")
ACTIONS = ("H", "M", "A")
METHODS = (
    "vanilla_pg",
    "action_dependent_cost_advantage",
    "full_ace",
    "scalar_only_valid_shift",
    "shuffled_validity_ace",
    "always_on_actor_regularization",
)
DISPLAY_NAMES = {
    "vanilla_pg": "Vanilla PG",
    "action_dependent_cost_advantage": "Action-dependent cost advantage",
    "full_ace": "Full ACE",
    "scalar_only_valid_shift": "Scalar-only valid shift",
    "shuffled_validity_ace": "Shuffled-validity ACE",
    "always_on_actor_regularization": "Always-on actor regularization",
}
SHORT_NAMES = {
    "vanilla_pg": "Vanilla PG",
    "action_dependent_cost_advantage": "Cost advantage",
    "full_ace": "Full ACE",
    "scalar_only_valid_shift": "Scalar shift",
    "shuffled_validity_ace": "Shuffled mask",
    "always_on_actor_regularization": "Always-on",
}
CONTRAST_SPECS = (
    ("C0", "full_ace", "vanilla_pg", "q_V", True),
    ("C1", "full_ace", "scalar_only_valid_shift", "q_V", True),
    ("C2", "full_ace", "shuffled_validity_ace", "q_V", True),
    ("C3", "shuffled_validity_ace", "full_ace", "q_X", True),
    ("C4", "full_ace", "shuffled_validity_ace", "U", False),
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(1000, 1050))
    updates: int = 400
    batch_size: int = 384
    learning_rate: float = 0.08
    initial_logit_sd: float = 0.15
    reward_noise_sd: float = 0.10
    actor_alpha: float = 0.30
    scalar_shift: float = 0.40
    action_cost_weight: float = 0.20
    logit_guard: float = 10.0
    record_every: int = 10
    invariance_tolerance: float = 1e-12
    required_sign_count: int = 45
    familywise_alpha: float = 0.05


@dataclass(frozen=True)
class SeedStream:
    initial_logits: np.ndarray
    contexts: np.ndarray
    action_uniforms: np.ndarray
    reward_shocks: np.ndarray
    shuffle_permutations: np.ndarray


UTILITY = np.asarray(
    [
        [1.00, 0.97, 0.94],
        [1.00, 0.35, 0.05],
        [1.00, 0.55, 0.20],
    ],
    dtype=np.float64,
)
CONTEXT_PROBABILITIES = np.asarray([0.50, 0.25, 0.25], dtype=np.float64)
ACTION_COSTS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--formal",
        action="store_true",
        help="Run the frozen 50-seed scientific configuration.",
    )
    mode.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="Run only seeds -2 and -1 for 10 updates; no scientific inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (defaults to a dedicated mode-specific subdirectory).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal summary.")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Replay once in an isolated directory and compare core artifacts byte for byte.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if rows:
        fieldnames = list(rows[0]) if fieldnames is None else fieldnames
    elif fieldnames is None:
        raise ValueError(f"fieldnames required for empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def make_seed_stream(seed: int, cfg: Config) -> SeedStream:
    # NumPy rejects negative integer seeds; map only the protocol-reserved smoke
    # labels into uint32 space while leaving every formal seed unchanged.
    rng_seed = seed if seed >= 0 else (1 << 32) + seed
    rng = np.random.default_rng(rng_seed)
    initial_logits = rng.normal(0.0, cfg.initial_logit_sd, size=(3, 3)).astype(
        np.float64
    )
    context_uniforms = rng.random((cfg.updates, cfg.batch_size))
    contexts = np.where(
        context_uniforms < 0.50,
        0,
        np.where(context_uniforms < 0.75, 1, 2),
    ).astype(np.int8)
    action_uniforms = rng.random((cfg.updates, cfg.batch_size))
    reward_shocks = rng.normal(
        0.0, cfg.reward_noise_sd, size=(cfg.updates, cfg.batch_size)
    ).astype(np.float64)
    permutation_keys = rng.random((cfg.updates, cfg.batch_size))
    shuffle_permutations = np.argsort(
        permutation_keys, axis=1, kind="stable"
    ).astype(np.int32)
    return SeedStream(
        initial_logits=initial_logits,
        contexts=contexts,
        action_uniforms=action_uniforms,
        reward_shocks=reward_shocks,
        shuffle_permutations=shuffle_permutations,
    )


def exact_endpoints(logits: np.ndarray) -> dict[str, float]:
    probabilities = softmax(logits)
    q_values = probabilities[:, 1:].sum(axis=1)
    context_utilities = np.sum(probabilities * UTILITY, axis=1)
    result: dict[str, float] = {}
    for context_index, context in enumerate(CONTEXTS):
        for action_index, action in enumerate(ACTIONS):
            result[f"logit_{context}_{action}"] = float(
                logits[context_index, action_index]
            )
            result[f"p_{context}_{action}"] = float(
                probabilities[context_index, action_index]
            )
        q_context = float(q_values[context_index])
        result[f"q_{context}"] = q_context
        result[f"aggressive_share_{context}"] = float(
            probabilities[context_index, 2] / q_context
        )
        result[f"utility_{context}"] = float(context_utilities[context_index])
    q_x = 0.5 * (q_values[1] + q_values[2])
    result["q_X"] = float(q_x)
    result["U"] = float(CONTEXT_PROBABILITIES @ context_utilities)
    result["selectivity_gap"] = float(q_values[0] - q_x)
    return result


def sample_actions(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    sample_cumulative = cumulative[:, :2]
    return np.sum(uniforms[:, None] > sample_cumulative, axis=1).astype(np.int8)


def financial_returns(
    method: str,
    contexts: np.ndarray,
    actions: np.ndarray,
    reward_shocks: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    returns = UTILITY[contexts, actions] + reward_shocks
    if method == "scalar_only_valid_shift":
        returns = returns - cfg.scalar_shift * (contexts == 0).astype(np.float64)
    elif method == "action_dependent_cost_advantage":
        returns = returns - cfg.action_cost_weight * ACTION_COSTS[contexts, actions]
    return returns


def response_set_log_gradient(probability: np.ndarray) -> np.ndarray:
    response_mass = float(probability[1] + probability[2])
    target = np.asarray(
        [0.0, probability[1] / response_mass, probability[2] / response_mass],
        dtype=np.float64,
    )
    return target - probability


def batch_gradients(
    method: str,
    logits: np.ndarray,
    contexts: np.ndarray,
    action_uniforms: np.ndarray,
    reward_shocks: np.ndarray,
    shuffle_permutation: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    probabilities = softmax(logits)
    sample_probabilities = probabilities[contexts]
    actions = sample_actions(sample_probabilities, action_uniforms)
    returns = financial_returns(method, contexts, actions, reward_shocks, cfg)
    one_hot_actions = np.eye(3, dtype=np.float64)[actions]
    financial_gradient = np.zeros((3, 3), dtype=np.float64)

    for context_index in range(3):
        context_mask = contexts == context_index
        if not np.any(context_mask):
            continue
        context_returns = returns[context_mask]
        centered_advantages = context_returns - context_returns.mean()
        scores = one_hot_actions[context_mask] - probabilities[context_index]
        financial_gradient[context_index] = np.sum(
            centered_advantages[:, None] * scores, axis=0
        ) / cfg.batch_size

    canonical_validity = contexts == 0
    if method == "full_ace":
        active_audit = canonical_validity
    elif method == "shuffled_validity_ace":
        active_audit = canonical_validity[shuffle_permutation]
    elif method == "always_on_actor_regularization":
        active_audit = np.ones(cfg.batch_size, dtype=bool)
    else:
        active_audit = np.zeros(cfg.batch_size, dtype=bool)

    audit_gradient = np.zeros((3, 3), dtype=np.float64)
    for context_index in range(3):
        active_count = int(np.sum(active_audit & (contexts == context_index)))
        if active_count:
            audit_gradient[context_index] = (
                cfg.actor_alpha
                * active_count
                / cfg.batch_size
                * response_set_log_gradient(probabilities[context_index])
            )

    counts = {
        "canonical_valid_rows": int(canonical_validity.sum()),
        "active_audit_rows": int(active_audit.sum()),
        "active_audit_rows_V": int(np.sum(active_audit & (contexts == 0))),
        "active_audit_rows_I": int(np.sum(active_audit & (contexts == 1))),
        "active_audit_rows_P": int(np.sum(active_audit & (contexts == 2))),
    }
    return financial_gradient, audit_gradient, counts


def checkpoint_row(
    seed: int,
    method: str,
    update: int,
    logits: np.ndarray,
    last_norms: dict[str, float],
) -> dict:
    return {
        "seed": seed,
        "method": method,
        "update": update,
        **exact_endpoints(logits),
        **last_norms,
    }


def train_method(
    seed: int, method: str, stream: SeedStream, cfg: Config
) -> tuple[list[dict], list[dict], dict]:
    logits = stream.initial_logits.copy()
    trajectories: list[dict] = []
    diagnostics: list[dict] = []
    last_norms = {
        "last_financial_gradient_norm": 0.0,
        "last_audit_gradient_norm": 0.0,
        "last_total_gradient_norm": 0.0,
    }
    trajectories.append(checkpoint_row(seed, method, 0, logits, last_norms))

    for update_index in range(cfg.updates):
        financial_gradient, audit_gradient, counts = batch_gradients(
            method=method,
            logits=logits,
            contexts=stream.contexts[update_index],
            action_uniforms=stream.action_uniforms[update_index],
            reward_shocks=stream.reward_shocks[update_index],
            shuffle_permutation=stream.shuffle_permutations[update_index],
            cfg=cfg,
        )
        total_gradient = financial_gradient + audit_gradient
        last_norms = {
            "last_financial_gradient_norm": float(np.linalg.norm(financial_gradient)),
            "last_audit_gradient_norm": float(np.linalg.norm(audit_gradient)),
            "last_total_gradient_norm": float(np.linalg.norm(total_gradient)),
        }
        diagnostics.append(
            {
                "seed": seed,
                "method": method,
                "update": update_index + 1,
                **counts,
                "dose_difference": counts["active_audit_rows"]
                - counts["canonical_valid_rows"],
                "financial_gradient_norm": last_norms[
                    "last_financial_gradient_norm"
                ],
                "audit_gradient_norm": last_norms["last_audit_gradient_norm"],
                "total_gradient_norm": last_norms["last_total_gradient_norm"],
                "financial_gradient_norm_V": float(
                    np.linalg.norm(financial_gradient[0])
                ),
                "financial_gradient_norm_I": float(
                    np.linalg.norm(financial_gradient[1])
                ),
                "financial_gradient_norm_P": float(
                    np.linalg.norm(financial_gradient[2])
                ),
                "audit_gradient_norm_V": float(np.linalg.norm(audit_gradient[0])),
                "audit_gradient_norm_I": float(np.linalg.norm(audit_gradient[1])),
                "audit_gradient_norm_P": float(np.linalg.norm(audit_gradient[2])),
            }
        )
        logits = np.clip(
            logits + cfg.learning_rate * total_gradient,
            -cfg.logit_guard,
            cfg.logit_guard,
        )
        completed_update = update_index + 1
        if completed_update % cfg.record_every == 0:
            trajectories.append(
                checkpoint_row(
                    seed, method, completed_update, logits, last_norms
                )
            )

    endpoints = exact_endpoints(logits)
    final = {
        "seed": seed,
        "method": method,
        **{
            f"initial_logit_{context}_{action}": float(
                stream.initial_logits[context_index, action_index]
            )
            for context_index, context in enumerate(CONTEXTS)
            for action_index, action in enumerate(ACTIONS)
        },
        **endpoints,
        "maximum_probability_sum_error": float(
            np.max(np.abs(softmax(logits).sum(axis=1) - 1.0))
        ),
        "finite_run": int(np.all(np.isfinite(logits))),
    }
    return trajectories, diagnostics, final


SUMMARY_METRICS = (
    "q_V",
    "q_I",
    "q_P",
    "q_X",
    "U",
    "selectivity_gap",
    "p_V_H",
    "p_V_M",
    "p_V_A",
    "p_I_H",
    "p_I_M",
    "p_I_A",
    "p_P_H",
    "p_P_M",
    "p_P_A",
    "aggressive_share_V",
    "aggressive_share_I",
    "aggressive_share_P",
)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    if n < 2:
        raise ValueError("at least two seeds are required for a summary")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    half_width = critical * sd / math.sqrt(n)
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_methods(finals: list[dict], cfg: Config) -> list[dict]:
    summaries: list[dict] = []
    for method in METHODS:
        rows = [row for row in finals if row["method"] == method]
        summary: dict[str, str | int | float] = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "n_seeds": len(rows),
        }
        for metric in SUMMARY_METRICS:
            metric_values = np.asarray(
                [float(row[metric]) for row in rows], dtype=np.float64
            )
            for statistic, value in summarize_values(metric_values).items():
                summary[f"{metric}_{statistic}"] = value
        summaries.append(summary)
    return summaries


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64), kind="stable")
    adjusted = np.zeros(len(p_values), dtype=np.float64)
    running_maximum = 0.0
    family_size = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = (family_size - rank) * p_values[int(original_index)]
        running_maximum = max(running_maximum, candidate)
        adjusted[int(original_index)] = min(1.0, running_maximum)
    return [float(value) for value in adjusted]


def compute_contrasts(
    finals: list[dict], cfg: Config, formal: bool
) -> tuple[list[dict], list[dict]]:
    index = {
        (int(row["seed"]), str(row["method"])): row for row in finals
    }
    contrast_rows: list[dict] = []
    difference_rows: list[dict] = []
    raw_p_values: list[float] = []
    for contrast_id, minuend, subtrahend, metric, sign_gate in CONTRAST_SPECS:
        differences = np.asarray(
            [
                float(index[(seed, minuend)][metric])
                - float(index[(seed, subtrahend)][metric])
                for seed in cfg.seeds
            ],
            dtype=np.float64,
        )
        for seed, difference in zip(cfg.seeds, differences, strict=True):
            difference_rows.append(
                {
                    "contrast": contrast_id,
                    "seed": seed,
                    "minuend_method": minuend,
                    "subtrahend_method": subtrahend,
                    "metric": metric,
                    "difference": float(difference),
                    "expected_sign": "positive",
                    "expected_sign_pass": int(difference > 0.0),
                }
            )
        statistics = summarize_values(differences)
        test = stats.ttest_1samp(differences, popmean=0.0, alternative="two-sided")
        raw_p_value = float(test.pvalue)
        raw_p_values.append(raw_p_value)
        contrast_rows.append(
            {
                "contrast": contrast_id,
                "minuend_method": minuend,
                "subtrahend_method": subtrahend,
                "metric": metric,
                "expected_sign": "positive",
                "n_pairs": len(differences),
                **statistics,
                "p_value_two_sided": raw_p_value,
                "holm_adjusted_p_value": 0.0,
                "expected_sign_count": int(np.sum(differences > 0.0)),
                "sign_count_gate_applies": int(sign_gate),
                "ci_gate_pass": "",
                "holm_gate_pass": "",
                "sign_count_gate_pass": "",
                "contrast_gate_pass": "",
            }
        )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(contrast_rows, adjusted_p_values, strict=True):
        row["holm_adjusted_p_value"] = adjusted_p
        if formal:
            ci_pass = float(row["ci95_low"]) > 0.0
            holm_pass = adjusted_p < cfg.familywise_alpha
            sign_pass = (
                int(row["expected_sign_count"]) >= cfg.required_sign_count
                if int(row["sign_count_gate_applies"])
                else True
            )
            row["ci_gate_pass"] = int(ci_pass)
            row["holm_gate_pass"] = int(holm_pass)
            row["sign_count_gate_pass"] = int(sign_pass)
            row["contrast_gate_pass"] = int(ci_pass and holm_pass and sign_pass)
    return contrast_rows, difference_rows


def validate_results(
    trajectories: list[dict],
    diagnostics: list[dict],
    finals: list[dict],
    failures: list[dict],
    contrasts: list[dict],
    cfg: Config,
    formal: bool,
) -> dict:
    errors: list[str] = []
    expected_runs = len(cfg.seeds) * len(METHODS)
    expected_checkpoints = cfg.updates // cfg.record_every + 1
    if failures:
        errors.append(f"{len(failures)} seed-method exceptions")
    if len(finals) != expected_runs:
        errors.append(f"completed {len(finals)} of {expected_runs} runs")
    if len(trajectories) != expected_runs * expected_checkpoints:
        errors.append(
            f"stored {len(trajectories)} of {expected_runs * expected_checkpoints} checkpoints"
        )
    if len(diagnostics) != expected_runs * cfg.updates:
        errors.append(
            f"stored {len(diagnostics)} of {expected_runs * cfg.updates} update diagnostics"
        )

    for row in finals:
        numeric_values = [
            float(value)
            for key, value in row.items()
            if key not in {"method"}
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            errors.append(
                f"non-finite endpoint seed={row['seed']} method={row['method']}"
            )
        if int(row["finite_run"]) != 1:
            errors.append(
                f"non-finite logits seed={row['seed']} method={row['method']}"
            )
        if float(row["maximum_probability_sum_error"]) > 1e-12:
            errors.append(
                f"invalid probability sum seed={row['seed']} method={row['method']}"
            )
        for context in CONTEXTS:
            for action in ACTIONS:
                probability = float(row[f"p_{context}_{action}"])
                if not 0.0 <= probability <= 1.0:
                    errors.append(
                        f"invalid probability seed={row['seed']} method={row['method']} "
                        f"context={context} action={action}"
                    )

    trajectory_index = {
        (int(row["seed"]), str(row["method"]), int(row["update"])): row
        for row in trajectories
    }
    vanilla_scalar_maximum = 0.0
    full_invalid_maximum = 0.0
    checkpoint_updates = range(0, cfg.updates + 1, cfg.record_every)
    all_logit_probability_keys = [
        f"{prefix}_{context}_{action}"
        for prefix in ("logit", "p")
        for context in CONTEXTS
        for action in ACTIONS
    ] + ["q_V", "q_I", "q_P", "q_X"]
    invalid_keys = [
        f"{prefix}_{context}_{action}"
        for prefix in ("logit", "p")
        for context in ("I", "P")
        for action in ACTIONS
    ] + ["q_I", "q_P", "q_X"]
    for seed in cfg.seeds:
        for update in checkpoint_updates:
            vanilla = trajectory_index.get((seed, "vanilla_pg", update))
            scalar = trajectory_index.get((seed, "scalar_only_valid_shift", update))
            full = trajectory_index.get((seed, "full_ace", update))
            if vanilla is None or scalar is None or full is None:
                errors.append(f"missing invariant checkpoint seed={seed} update={update}")
                continue
            for key in all_logit_probability_keys:
                vanilla_scalar_maximum = max(
                    vanilla_scalar_maximum,
                    abs(float(vanilla[key]) - float(scalar[key])),
                )
            for key in invalid_keys:
                full_invalid_maximum = max(
                    full_invalid_maximum,
                    abs(float(full[key]) - float(vanilla[key])),
                )

    vanilla_scalar_pass = vanilla_scalar_maximum <= cfg.invariance_tolerance
    full_invalid_pass = full_invalid_maximum <= cfg.invariance_tolerance
    if not vanilla_scalar_pass:
        errors.append(
            "vanilla/scalar invariant exceeded tolerance: "
            f"{vanilla_scalar_maximum:.17g}"
        )
    if not full_invalid_pass:
        errors.append(
            "full-ACE invalid-context invariant exceeded tolerance: "
            f"{full_invalid_maximum:.17g}"
        )

    shuffled_rows = [
        row for row in diagnostics if row["method"] == "shuffled_validity_ace"
    ]
    shuffled_dose_maximum = max(
        (abs(int(row["dose_difference"])) for row in shuffled_rows), default=0
    )
    shuffled_dose_pass = shuffled_dose_maximum == 0
    if not shuffled_dose_pass:
        errors.append(f"shuffled audit dose mismatch: {shuffled_dose_maximum}")

    structural_pass = not errors
    contrast_gate_pass = formal and all(
        int(row["contrast_gate_pass"]) == 1 for row in contrasts
    )
    formal_decision = (
        "GO" if structural_pass and contrast_gate_pass else "NO-GO"
    ) if formal else "ENGINEERING-SMOKE-ONLY"
    return {
        "run_mode": "formal" if formal else "engineering_smoke",
        "structural_pass": structural_pass,
        "structural_errors": errors,
        "expected_seed_method_runs": expected_runs,
        "completed_seed_method_runs": len(finals),
        "failure_count": len(failures),
        "maximum_vanilla_scalar_checkpoint_difference": vanilla_scalar_maximum,
        "vanilla_scalar_invariance_pass": vanilla_scalar_pass,
        "maximum_full_ace_vanilla_invalid_checkpoint_difference": full_invalid_maximum,
        "full_ace_invalid_invariance_pass": full_invalid_pass,
        "maximum_shuffled_dose_difference": shuffled_dose_maximum,
        "shuffled_dose_match_pass": shuffled_dose_pass,
        "all_contrast_gates_pass": bool(contrast_gate_pass),
        "scientific_decision": formal_decision,
        "claim_permitted": bool(formal and structural_pass and contrast_gate_pass),
    }


def format_ci(row: dict, metric: str, digits: int = 3) -> str:
    return (
        f"{row[f'{metric}_mean']:.{digits}f} "
        f"[{row[f'{metric}_ci95_low']:.{digits}f}, "
        f"{row[f'{metric}_ci95_high']:.{digits}f}]"
    )


def plot_results(
    summaries: list[dict], trajectories: list[dict], output_dir: Path
) -> None:
    positions = np.arange(len(METHODS), dtype=np.float64)
    q_v = [float(row["q_V_mean"]) for row in summaries]
    q_x = [float(row["q_X_mean"]) for row in summaries]
    q_v_error = [float(row["q_V_ci95_high"]) - mean for row, mean in zip(summaries, q_v, strict=True)]
    q_x_error = [float(row["q_X_ci95_high"]) - mean for row, mean in zip(summaries, q_x, strict=True)]
    utility = [float(row["U_mean"]) for row in summaries]
    utility_error = [float(row["U_ci95_high"]) - mean for row, mean in zip(summaries, utility, strict=True)]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        q_v,
        width,
        yerr=q_v_error,
        capsize=2.5,
        color="#3B6FB6",
        label=r"Valid $q_V$",
    )
    axes[0].bar(
        positions + width / 2,
        q_x,
        width,
        yerr=q_x_error,
        capsize=2.5,
        color="#D05A4E",
        label=r"Invalid $q_X$",
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Response-set probability")
    axes[0].set_title("Context-aligned response pressure")
    axes[0].set_xticks(positions, ["PG", "Cost", "Full", "Shift", "Shuffle", "Always"])
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(positions, utility, yerr=utility_error, capsize=2.5, color="#5B8E7D")
    utility_floor = min(utility) - 0.02
    axes[1].set_ylim(max(0.0, utility_floor), min(1.01, max(utility) + 0.02))
    axes[1].set_ylabel("Exact expected utility")
    axes[1].set_title("Financial guardrail (not a ranking)")
    axes[1].set_xticks(positions, ["PG", "Cost", "Full", "Shift", "Shuffle", "Always"])

    curve_methods = ("vanilla_pg", "full_ace", "shuffled_validity_ace")
    curve_colors = {
        "vanilla_pg": "#555555",
        "full_ace": "#3B6FB6",
        "shuffled_validity_ace": "#D05A4E",
    }
    for method in curve_methods:
        method_rows = [row for row in trajectories if row["method"] == method]
        updates = sorted({int(row["update"]) for row in method_rows})
        means = [
            float(np.mean([float(row["selectivity_gap"]) for row in method_rows if int(row["update"]) == update]))
            for update in updates
        ]
        axes[2].plot(
            updates,
            means,
            linewidth=1.8,
            color=curve_colors[method],
            label=SHORT_NAMES[method],
        )
    axes[2].axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
    axes[2].set_xlabel("Update")
    axes[2].set_ylabel(r"Selectivity $q_V-q_X$")
    axes[2].set_title("Selectivity trajectory")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "main_experiment.png", dpi=220)
    fig.savefig(
        output_dir / "main_experiment.pdf",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "ACE-RL E-main"},
    )
    plt.close(fig)


def write_tex_table(
    summaries: list[dict], validation: dict, formal: bool, output_dir: Path
) -> None:
    if formal:
        decision_text = (
            "All frozen gates pass."
            if validation["scientific_decision"] == "GO"
            else "At least one frozen gate fails; the result is a pre-specified NO-GO."
        )
    else:
        decision_text = "Engineering smoke only; these two negative seeds are not scientific results."
    lines = [
        "% Auto-generated by run_main_experiment.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Frozen multi-context set-valued audit assay. " + decision_text + " Values are seed means with two-sided 95\\% t intervals; utility is a guardrail, not a ranking claim.}",
        "\\label{tab:multi_context_audit_main}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & $q_V$ & $q_X$ & $q_V-q_X$ & Exact utility $U$ \\\\",
        "\\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{SHORT_NAMES[str(row['method'])]} & {format_ci(row, 'q_V')} & "
            f"{format_ci(row, 'q_X')} & {format_ci(row, 'selectivity_gap')} & "
            f"{format_ci(row, 'U')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (output_dir / "table_snippet.tex").write_text("\n".join(lines), encoding="utf-8")


def write_results(
    summaries: list[dict], contrasts: list[dict], validation: dict, formal: bool, output_dir: Path
) -> None:
    lines = [
        "# E-main: multi-context set-valued audit assay",
        "",
        f"Run mode: **{validation['run_mode']}**. Decision: **{validation['scientific_decision']}**.",
        "",
    ]
    if not formal:
        lines.extend(
            [
                "This run uses only the protocol-authorized engineering seeds -2 and -1 for 10 updates. It tests file production, finite updates, dose matching, and structural invariants only. Its endpoint values and inferential statistics must not be cited as scientific evidence or used for tuning.",
                "",
            ]
        )
    lines.extend(
        [
            "## Endpoints",
            "",
            "| Method | q_V mean [95% CI] | q_X mean [95% CI] | Selectivity mean [95% CI] | U mean [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['display_name']} | {format_ci(row, 'q_V')} | {format_ci(row, 'q_X')} | "
            f"{format_ci(row, 'selectivity_gap')} | {format_ci(row, 'U')} |"
        )
    lines.extend(
        [
            "",
            "## Frozen contrasts",
            "",
            "| Contrast | Metric | Mean difference | 95% CI | Raw p | Holm p | Positive pairs | Gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        gate = row["contrast_gate_pass"] if formal else "not evaluated"
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['mean']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['p_value_two_sided']:.3e} | {row['holm_adjusted_p_value']:.3e} | "
            f"{row['expected_sign_count']}/{row['n_pairs']} | {gate} |"
        )
    lines.extend(
        [
            "",
            "## Structural checks",
            "",
            f"- Complete finite runs: {validation['completed_seed_method_runs']}/{validation['expected_seed_method_runs']}; failures: {validation['failure_count']}.",
            f"- Vanilla/scalar maximum checkpoint difference: {validation['maximum_vanilla_scalar_checkpoint_difference']:.3e}.",
            f"- Full-ACE/vanilla maximum invalid-context checkpoint difference: {validation['maximum_full_ace_vanilla_invalid_checkpoint_difference']:.3e}.",
            f"- Maximum shuffled-mask dose difference: {validation['maximum_shuffled_dose_difference']} rows.",
            "",
            "## Claim boundary",
            "",
            "This controlled assay cannot establish limit-order-book effectiveness, collusion mitigation, profit improvement, equilibrium selection, deployment safety, legal interpretation, robust target attainment, or superiority over action-dependent constrained learning. It does not replace or repair the earlier frozen two-context NO-GO.",
            "",
        ]
    )
    if formal and validation["claim_permitted"]:
        lines.extend(
            [
                "Permitted statement: In a frozen controlled quote-choice assay, set-valued audit labels provide action direction, a dose-matched validity mask concentrates pressure on admissible contexts, and parameter separation prevents audit-to-critic leakage. The final clause additionally requires the separate A3 gate.",
                "",
            ]
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


CORE_ARTIFACTS = (
    "failures.csv",
    "learning_curves.csv",
    "update_diagnostics.csv",
    "seed_results.csv",
    "paired_differences.csv",
    "paired_contrasts.csv",
    "summary.csv",
    "summary.json",
    "main_experiment.pdf",
    "main_experiment.png",
    "table_snippet.tex",
    "RESULTS.md",
)


def write_manifest_and_checksums(
    output_dir: Path,
    source_dir: Path,
    cfg: Config,
    validation: dict,
    run_mode: str,
    command: str,
    duration_seconds: float,
) -> None:
    hashes = {
        "PROTOCOL.md": sha256(source_dir / "PROTOCOL.md"),
        "run_main_experiment.py": sha256(source_dir / "run_main_experiment.py"),
    }
    hashes.update({name: sha256(output_dir / name) for name in CORE_ARTIFACTS})
    manifest = {
        "status": "complete" if validation["structural_pass"] else "failed",
        "run_mode": run_mode,
        "scientific_decision": validation["scientific_decision"],
        "claim_permitted": validation["claim_permitted"],
        "claim_boundary": (
            "controlled quote-choice mechanism assay only; not LOB effectiveness, "
            "collusion mitigation, financial dominance, or deployment evidence"
        ),
        "protocol_status": "frozen before implementation and first run",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "command": command,
        "duration_seconds": duration_seconds,
        "config": asdict(cfg),
        "contexts": list(CONTEXTS),
        "context_probabilities": CONTEXT_PROBABILITIES.tolist(),
        "actions": list(ACTIONS),
        "financial_utility_matrix": UTILITY.tolist(),
        "action_cost_matrix": ACTION_COSTS.tolist(),
        "methods": list(METHODS),
        "validation": validation,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "sha256": hashes,
    }
    write_json(output_dir / "manifest.json", manifest)
    # The ledger is rooted in the artifact directory, so include only files
    # physically present there. Source hashes remain explicit in the manifest.
    checksum_entries = {
        name: digest for name, digest in hashes.items() if (output_dir / name).is_file()
    }
    checksum_entries["manifest.json"] = sha256(output_dir / "manifest.json")
    (output_dir / "checksums.sha256").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(checksum_entries.items())
        ),
        encoding="utf-8",
    )


def update_manifest_after_replay(output_dir: Path, receipt: dict) -> None:
    receipt_path = output_dir / "determinism_receipt.json"
    write_json(receipt_path, receipt)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deterministic_replay"] = {
        "status": receipt["status"],
        "compared_artifact_count": receipt["compared_artifact_count"],
        "matched_artifact_count": receipt["matched_artifact_count"],
        "mismatched_artifacts": receipt["mismatched_artifacts"],
    }
    manifest["sha256"]["determinism_receipt.json"] = sha256(receipt_path)
    write_json(manifest_path, manifest)
    checksum_entries = {
        name: digest
        for name, digest in manifest["sha256"].items()
        if (output_dir / name).is_file()
    }
    checksum_entries["manifest.json"] = sha256(manifest_path)
    (output_dir / "checksums.sha256").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(checksum_entries.items())
        ),
        encoding="utf-8",
    )


def verify_determinism(
    output_dir: Path, source_dir: Path, formal: bool
) -> dict:
    replay_dir = Path(tempfile.mkdtemp(prefix="ace_multi_context_replay_"))
    try:
        mode_flag = "--formal" if formal else "--engineering-smoke"
        command = [
            sys.executable,
            str(source_dir / "run_main_experiment.py"),
            mode_flag,
            "--output-dir",
            str(replay_dir),
            "--quiet",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "single deterministic replay failed without retry: "
                f"returncode={completed.returncode}; stderr={completed.stderr.strip()}"
            )
        mismatches = [
            name
            for name in CORE_ARTIFACTS
            if (output_dir / name).read_bytes() != (replay_dir / name).read_bytes()
        ]
        receipt = {
            "status": "PASS" if not mismatches else "FAIL",
            "verification": (
                "byte-for-byte comparison of core scientific artifacts against "
                "one isolated temporary-directory execution"
            ),
            "replay_command": (
                f"python3 run_main_experiment.py {mode_flag} "
                "--output-dir <temporary-directory> --quiet"
            ),
            "compared_artifact_count": len(CORE_ARTIFACTS),
            "matched_artifact_count": len(CORE_ARTIFACTS) - len(mismatches),
            "mismatched_artifacts": mismatches,
            "comparison_sha256": {
                name: sha256(output_dir / name) for name in CORE_ARTIFACTS
            },
        }
        update_manifest_after_replay(output_dir, receipt)
        if mismatches:
            raise RuntimeError(
                "deterministic replay mismatched: " + ", ".join(mismatches)
            )
        return receipt
    finally:
        shutil.rmtree(replay_dir)


def run(args: argparse.Namespace) -> int:
    source_dir = Path(__file__).resolve().parent
    protocol_path = source_dir / "PROTOCOL.md"
    actual_protocol_hash = sha256(protocol_path)
    if actual_protocol_hash != EXPECTED_PROTOCOL_SHA256:
        print(
            "ERROR: PROTOCOL.md hash mismatch; refusing to run. "
            f"expected={EXPECTED_PROTOCOL_SHA256} actual={actual_protocol_hash}",
            file=sys.stderr,
        )
        return 2

    formal = bool(args.formal)
    frozen_cfg = Config()
    cfg = (
        frozen_cfg
        if formal
        else replace(frozen_cfg, seeds=(-2, -1), updates=10)
    )
    if not formal and (cfg.seeds != (-2, -1) or cfg.updates > 10):
        raise AssertionError("engineering smoke exceeded the frozen allowance")
    if cfg.updates % cfg.record_every != 0:
        raise AssertionError("updates must align with curve checkpoints")

    if args.output_dir is None:
        output_dir = source_dir / ("results_main" if formal else "engineering_smoke")
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trajectories: list[dict] = []
    diagnostics: list[dict] = []
    finals: list[dict] = []
    failures: list[dict] = []

    for seed in cfg.seeds:
        stream = make_seed_stream(seed, cfg)
        for method in METHODS:
            try:
                method_trajectories, method_diagnostics, final = train_method(
                    seed, method, stream, cfg
                )
                trajectories.extend(method_trajectories)
                diagnostics.extend(method_diagnostics)
                finals.append(final)
            except Exception as exc:  # pragma: no cover - retained as failure ledger
                failures.append(
                    {
                        "seed": seed,
                        "method": method,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    write_csv(output_dir / "learning_curves.csv", trajectories)
    write_csv(output_dir / "update_diagnostics.csv", diagnostics)
    write_csv(output_dir / "seed_results.csv", finals)
    write_csv(
        output_dir / "failures.csv",
        failures,
        fieldnames=["seed", "method", "exception_type", "message"],
    )
    expected_runs = len(cfg.seeds) * len(METHODS)
    if len(finals) != expected_runs or failures:
        print(
            "ERROR: incomplete E-main run; raw outputs and failure ledger were written. "
            "No automatic retry was attempted.",
            file=sys.stderr,
        )
        return 2

    summaries = summarize_methods(finals, cfg)
    contrasts, paired_differences = compute_contrasts(finals, cfg, formal)
    validation = validate_results(
        trajectories,
        diagnostics,
        finals,
        failures,
        contrasts,
        cfg,
        formal,
    )
    write_csv(output_dir / "paired_differences.csv", paired_differences)
    write_csv(output_dir / "paired_contrasts.csv", contrasts)
    write_csv(output_dir / "summary.csv", summaries)
    write_json(
        output_dir / "summary.json",
        {
            "run_mode": "formal" if formal else "engineering_smoke",
            "claim_boundary": "frozen controlled quote-choice mechanism assay only",
            "config": asdict(cfg),
            "methods": summaries,
            "contrasts": contrasts,
            "validation": validation,
        },
    )
    plot_results(summaries, trajectories, output_dir)
    write_tex_table(summaries, validation, formal, output_dir)
    write_results(summaries, contrasts, validation, formal, output_dir)
    duration_seconds = round(time.perf_counter() - started, 6)
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    write_manifest_and_checksums(
        output_dir,
        source_dir,
        cfg,
        validation,
        "formal" if formal else "engineering_smoke",
        command,
        duration_seconds,
    )

    replay_receipt = None
    if args.verify_determinism:
        replay_receipt = verify_determinism(output_dir, source_dir, formal)

    if not args.quiet:
        print(f"Run mode: {validation['run_mode']}")
        print(f"Decision: {validation['scientific_decision']}")
        print(f"Completed runs: {len(finals)}/{expected_runs}; failures={len(failures)}")
        print(
            "Structural invariants: "
            f"scalar={validation['maximum_vanilla_scalar_checkpoint_difference']:.3e}, "
            f"invalid={validation['maximum_full_ace_vanilla_invalid_checkpoint_difference']:.3e}, "
            f"dose={validation['maximum_shuffled_dose_difference']}"
        )
        if replay_receipt is not None:
            print(
                "Deterministic replay: "
                f"{replay_receipt['status']} "
                f"({replay_receipt['matched_artifact_count']}/"
                f"{replay_receipt['compared_artifact_count']} byte-identical)"
            )
        print(f"Artifacts: {output_dir}")
    return 0 if validation["structural_pass"] else 2


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
