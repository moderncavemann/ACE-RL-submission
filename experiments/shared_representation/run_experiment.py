#!/usr/bin/env python3
"""Frozen shared-MLP contextual-selectivity assay for ACE-RL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

EXPECTED_PROTOCOL_SHA256 = "658f35fc3a2b70b65b935790ea47416a666292c86aeda19dc3611026d9aa2ff6"
METHODS = (
    "vanilla_shared",
    "full_ace_shared",
    "shuffled_validity_shared",
    "always_on_shared",
)
DISPLAY = {
    "vanilla_shared": "Vanilla shared MLP",
    "full_ace_shared": "Full ACE (shared MLP)",
    "shuffled_validity_shared": "Shuffled validity",
    "always_on_shared": "Always-on",
}
CONTEXTS = ("V", "I", "P")
ACTIONS = ("H", "M", "A")
CONTEXT_CENTERS = np.asarray(
    [[1.0, 0.0, 0.0], [0.8, 1.0, 0.2], [0.8, 0.2, 1.0]], dtype=np.float64
)
UTILITY = np.asarray(
    [[1.00, 0.97, 0.94], [1.00, 0.35, 0.05], [1.00, 0.55, 0.20]],
    dtype=np.float64,
)
CONTEXT_PROBABILITIES = np.asarray([0.50, 0.25, 0.25], dtype=np.float64)
CONTRASTS = (
    ("S0", "full_ace_shared", "vanilla_shared", "q_V"),
    ("S1", "shuffled_validity_shared", "full_ace_shared", "q_X"),
    ("S2", "always_on_shared", "full_ace_shared", "q_X"),
    ("S3", "full_ace_shared", "shuffled_validity_shared", "selectivity"),
    ("S4", "full_ace_shared", "shuffled_validity_shared", "U"),
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(3000, 3100))
    updates: int = 400
    batch_size: int = 384
    hidden_width: int = 8
    learning_rate: float = 0.03
    audit_alpha: float = 0.30
    observation_noise_sd: float = 0.08
    reward_noise_sd: float = 0.10
    record_every: int = 10
    parameter_guard: float = 10.0
    sign_threshold: int = 90
    spillover_threshold: float = 1e-6
    spillover_count_threshold: int = 95
    familywise_alpha: float = 0.05


@dataclass
class Parameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def copy(self) -> "Parameters":
        return Parameters(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())


@dataclass(frozen=True)
class SeedStream:
    initial: Parameters
    contexts: np.ndarray
    observations: np.ndarray
    action_uniforms: np.ndarray
    reward_shocks: np.ndarray
    shuffle_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def rng_seed(seed: int) -> int:
    return seed if seed >= 0 else (1 << 32) + seed


def make_stream(seed: int, cfg: Config) -> SeedStream:
    rng = np.random.default_rng(rng_seed(seed))
    initial = Parameters(
        w1=rng.normal(0.0, 0.25, size=(3, cfg.hidden_width)).astype(np.float64),
        b1=rng.normal(0.0, 0.05, size=cfg.hidden_width).astype(np.float64),
        w2=rng.normal(0.0, 0.20, size=(cfg.hidden_width, 3)).astype(np.float64),
        b2=rng.normal(0.0, 0.05, size=3).astype(np.float64),
    )
    context_uniforms = rng.random((cfg.updates, cfg.batch_size))
    contexts = np.where(context_uniforms < 0.50, 0, np.where(context_uniforms < 0.75, 1, 2)).astype(np.int8)
    observation_noise = rng.normal(
        0.0, cfg.observation_noise_sd, size=(cfg.updates, cfg.batch_size, 3)
    ).astype(np.float64)
    observations = CONTEXT_CENTERS[contexts] + observation_noise
    action_uniforms = rng.random((cfg.updates, cfg.batch_size)).astype(np.float64)
    reward_shocks = rng.normal(
        0.0, cfg.reward_noise_sd, size=(cfg.updates, cfg.batch_size)
    ).astype(np.float64)
    shuffle_keys = rng.random((cfg.updates, cfg.batch_size))
    shuffle_indices = np.argsort(shuffle_keys, axis=1, kind="stable").astype(np.int32)
    return SeedStream(initial, contexts, observations, action_uniforms, reward_shocks, shuffle_indices)


def forward(params: Parameters, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(observations @ params.w1 + params.b1)
    probabilities = softmax(hidden @ params.w2 + params.b2)
    return hidden, probabilities


def endpoints(params: Parameters) -> dict[str, float]:
    _, probabilities = forward(params, CONTEXT_CENTERS)
    response = probabilities[:, 1:].sum(axis=1)
    context_utility = np.sum(probabilities * UTILITY, axis=1)
    result: dict[str, float] = {}
    for c_index, context in enumerate(CONTEXTS):
        result[f"q_{context}"] = float(response[c_index])
        result[f"utility_{context}"] = float(context_utility[c_index])
        for a_index, action in enumerate(ACTIONS):
            result[f"p_{context}_{action}"] = float(probabilities[c_index, a_index])
    result["q_X"] = float(0.5 * (response[1] + response[2]))
    result["selectivity"] = float(response[0] - result["q_X"])
    result["U"] = float(CONTEXT_PROBABILITIES @ context_utility)
    return result


def sample_actions(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    return np.sum(uniforms[:, None] > np.cumsum(probabilities, axis=1)[:, :2], axis=1).astype(np.int8)


def centered_advantages(contexts: np.ndarray, returns: np.ndarray) -> np.ndarray:
    advantages = np.zeros_like(returns)
    for context in range(3):
        mask = contexts == context
        if np.any(mask):
            advantages[mask] = returns[mask] - np.mean(returns[mask])
    return advantages


def active_mask(method: str, contexts: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    canonical = contexts == 0
    if method == "full_ace_shared":
        return canonical.astype(np.float64)
    if method == "shuffled_validity_shared":
        return canonical[permutation].astype(np.float64)
    if method == "always_on_shared":
        return np.ones_like(contexts, dtype=np.float64)
    return np.zeros_like(contexts, dtype=np.float64)


def update_parameters(
    params: Parameters,
    observations: np.ndarray,
    contexts: np.ndarray,
    uniforms: np.ndarray,
    reward_shocks: np.ndarray,
    permutation: np.ndarray,
    method: str,
    cfg: Config,
) -> tuple[float, float]:
    hidden, probabilities = forward(params, observations)
    actions = sample_actions(probabilities, uniforms)
    returns = UTILITY[contexts, actions] + reward_shocks
    advantages = centered_advantages(contexts, returns)
    one_hot = np.eye(3, dtype=np.float64)[actions]
    batch_size = float(len(contexts))
    grad_logits = advantages[:, None] * (one_hot - probabilities) / batch_size

    mask = active_mask(method, contexts, permutation)
    if np.any(mask):
        response_mass = probabilities[:, 1:].sum(axis=1)
        conditional = np.zeros_like(probabilities)
        conditional[:, 1:] = probabilities[:, 1:] / response_mass[:, None]
        grad_logits += (
            cfg.audit_alpha * mask[:, None] * (conditional - probabilities) / batch_size
        )

    grad_w2 = hidden.T @ grad_logits
    grad_b2 = np.sum(grad_logits, axis=0)
    grad_hidden = (grad_logits @ params.w2.T) * (1.0 - hidden * hidden)
    grad_w1 = observations.T @ grad_hidden
    grad_b1 = np.sum(grad_hidden, axis=0)
    financial_norm = float(np.linalg.norm(advantages[:, None] * (one_hot - probabilities) / batch_size))
    audit_norm = float(np.linalg.norm(mask[:, None] * (probabilities - np.eye(3)[actions]) * 0.0))

    params.w1 += cfg.learning_rate * grad_w1
    params.b1 += cfg.learning_rate * grad_b1
    params.w2 += cfg.learning_rate * grad_w2
    params.b2 += cfg.learning_rate * grad_b2
    for array in (params.w1, params.b1, params.w2, params.b2):
        np.clip(array, -cfg.parameter_guard, cfg.parameter_guard, out=array)
    return financial_norm, audit_norm


def run_seed(seed: int, cfg: Config) -> tuple[list[dict], list[dict]]:
    stream = make_stream(seed, cfg)
    seed_rows: list[dict] = []
    curve_rows: list[dict] = []
    for method in METHODS:
        params = stream.initial.copy()
        start = endpoints(params)
        curve_rows.append({"seed": seed, "method": method, "update": 0, **start})
        for update in range(cfg.updates):
            update_parameters(
                params,
                stream.observations[update],
                stream.contexts[update],
                stream.action_uniforms[update],
                stream.reward_shocks[update],
                stream.shuffle_indices[update],
                method,
                cfg,
            )
            if (update + 1) % cfg.record_every == 0:
                curve_rows.append(
                    {"seed": seed, "method": method, "update": update + 1, **endpoints(params)}
                )
        final = endpoints(params)
        finite = all(math.isfinite(value) for value in final.values())
        probabilities_valid = all(
            0.0 <= final[key] <= 1.0 for key in final if key.startswith("p_") or key.startswith("q_")
        )
        seed_rows.append(
            {"seed": seed, "method": method, "finite": int(finite), "probabilities_valid": int(probabilities_valid), **final}
        )
    return seed_rows, curve_rows


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(values: np.ndarray) -> tuple[float, float, float, float, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(len(values))
    critical = float(stats.t.ppf(0.975, len(values) - 1))
    low, high = mean - critical * se, mean + critical * se
    if sd == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
    return mean, sd, float(low), float(high), p_value


def aggregate(seed_rows: list[dict], cfg: Config, formal: bool) -> tuple[list[dict], list[dict], dict]:
    by_method = {method: {int(row["seed"]): row for row in seed_rows if row["method"] == method} for method in METHODS}
    summary_rows: list[dict] = []
    for method in METHODS:
        rows = list(by_method[method].values())
        for metric in ("q_V", "q_X", "selectivity", "U"):
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            mean, sd, low, high, _ = paired_statistics(values)
            summary_rows.append(
                {"method": method, "metric": metric, "mean": mean, "sd": sd, "ci_low": low, "ci_high": high, "n": len(values)}
            )

    contrast_rows: list[dict] = []
    difference_rows: list[dict] = []
    p_values: list[float] = []
    for contrast, left, right, metric in CONTRASTS:
        differences = np.asarray(
            [by_method[left][seed][metric] - by_method[right][seed][metric] for seed in cfg.seeds], dtype=np.float64
        )
        mean, sd, low, high, p_value = paired_statistics(differences)
        p_values.append(p_value)
        sign_count = int(np.sum(differences > 0.0))
        contrast_rows.append(
            {"contrast": contrast, "left": left, "right": right, "metric": metric, "mean_difference": mean,
             "sd_difference": sd, "ci_low": low, "ci_high": high, "p_value": p_value,
             "positive_sign_count": sign_count, "n": len(differences)}
        )
        for seed, difference in zip(cfg.seeds, differences, strict=True):
            difference_rows.append({"seed": seed, "contrast": contrast, "metric": metric, "difference": float(difference)})
    adjusted = holm_adjust(p_values)
    for row, adjusted_p in zip(contrast_rows, adjusted, strict=True):
        row["holm_p_value"] = adjusted_p
        row["ci_gate"] = bool(row["ci_low"] > 0.0)
        row["holm_gate"] = bool(adjusted_p < cfg.familywise_alpha)
        row["sign_gate"] = bool(row["positive_sign_count"] >= cfg.sign_threshold)
        row["gate_pass"] = bool(row["ci_gate"] and row["holm_gate"] and row["sign_gate"])

    spillovers = np.asarray(
        [abs(by_method["full_ace_shared"][seed]["q_X"] - by_method["vanilla_shared"][seed]["q_X"]) for seed in cfg.seeds],
        dtype=np.float64,
    )
    spillover_count = int(np.sum(spillovers > cfg.spillover_threshold))
    integrity = {
        "expected_runs": len(cfg.seeds) * len(METHODS),
        "completed_runs": len(seed_rows),
        "finite_runs": int(sum(int(row["finite"]) for row in seed_rows)),
        "valid_probability_runs": int(sum(int(row["probabilities_valid"]) for row in seed_rows)),
    }
    gates = {
        "contrast_gates_pass": bool(all(row["gate_pass"] for row in contrast_rows)) if formal else None,
        "spillover_count": spillover_count,
        "spillover_required": cfg.spillover_count_threshold,
        "spillover_gate_pass": bool(spillover_count >= cfg.spillover_count_threshold) if formal else None,
    }
    all_integrity = all(value == integrity["expected_runs"] for key, value in integrity.items() if key != "expected_runs")
    scientific_go = bool(all_integrity and gates["contrast_gates_pass"] and gates["spillover_gate_pass"]) if formal else None
    payload = {"formal": formal, "integrity": integrity, "gates": gates, "scientific_decision": "GO" if scientific_go else ("NO-GO" if formal else "NOT_EVALUATED_SMOKE")}
    return summary_rows, contrast_rows, {"summary": payload, "differences": difference_rows, "spillovers": spillovers.tolist()}


def make_figure(curve_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), constrained_layout=True)
    metrics = (("q_V", "Valid response probability"), ("q_X", "Invalid response probability"), ("selectivity", "Valid-minus-invalid selectivity"))
    colors = {"vanilla_shared": "#5B6770", "full_ace_shared": "#0072B2", "shuffled_validity_shared": "#D55E00", "always_on_shared": "#CC79A7"}
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for method in METHODS:
            method_rows = [row for row in curve_rows if row["method"] == method]
            updates = sorted({int(row["update"]) for row in method_rows})
            means, lows, highs = [], [], []
            for update in updates:
                values = np.asarray([float(row[metric]) for row in method_rows if int(row["update"]) == update])
                mean, sd, low, high, _ = paired_statistics(values)
                means.append(mean); lows.append(low); highs.append(high)
            axis.plot(updates, means, label=DISPLAY[method], color=colors[method], linewidth=1.5)
            axis.fill_between(updates, lows, highs, color=colors[method], alpha=0.12, linewidth=0)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Update", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Probability", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(output_dir / "shared_selectivity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "shared_selectivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_table(path: Path, summary_rows: list[dict]) -> None:
    lookup = {(row["method"], row["metric"]): row for row in summary_rows}
    lines = ["% Generated by shared_representation_selectivity_v1/run_experiment.py", "\\begin{tabular}{lrrrr}", "\\toprule", "Method & $q_V$ & $q_X$ & $q_V-q_X$ & $U$\\\\", "\\midrule"]
    for method in METHODS:
        values = [lookup[(method, metric)]["mean"] for metric in ("q_V", "q_X", "selectivity", "U")]
        lines.append(f"{DISPLAY[method]} & " + " & ".join(f"{value:.3f}" for value in values) + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


CORE_ARTIFACTS = ("seed_results.csv", "learning_curves.csv", "paired_differences.csv", "paired_contrasts.csv", "summary.csv", "summary.json")


def execute(cfg: Config, output_dir: Path, formal: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    failures: list[dict] = []
    seed_rows: list[dict] = []
    curve_rows: list[dict] = []
    for seed in cfg.seeds:
        try:
            rows, curves = run_seed(seed, cfg)
            seed_rows.extend(rows); curve_rows.extend(curves)
        except Exception as exc:  # preserve every failure without retry
            failures.append({"seed": seed, "stage": "run_seed", "error_type": type(exc).__name__, "message": str(exc)})

    seed_fields = ["seed", "method", "finite", "probabilities_valid", "q_V", "utility_V", "p_V_H", "p_V_M", "p_V_A", "q_I", "utility_I", "p_I_H", "p_I_M", "p_I_A", "q_P", "utility_P", "p_P_H", "p_P_M", "p_P_A", "q_X", "selectivity", "U"]
    curve_fields = ["seed", "method", "update", "q_V", "utility_V", "p_V_H", "p_V_M", "p_V_A", "q_I", "utility_I", "p_I_H", "p_I_M", "p_I_A", "q_P", "utility_P", "p_P_H", "p_P_M", "p_P_A", "q_X", "selectivity", "U"]
    write_csv(output_dir / "seed_results.csv", seed_rows, seed_fields)
    write_csv(output_dir / "learning_curves.csv", curve_rows, curve_fields)
    write_csv(output_dir / "failures.csv", failures, ["seed", "stage", "error_type", "message"])
    summary_rows, contrast_rows, aggregate_payload = aggregate(seed_rows, cfg, formal)
    difference_rows = aggregate_payload.pop("differences")
    spillovers = aggregate_payload.pop("spillovers")
    write_csv(output_dir / "summary.csv", summary_rows, ["method", "metric", "mean", "sd", "ci_low", "ci_high", "n"])
    write_csv(output_dir / "paired_contrasts.csv", contrast_rows, ["contrast", "left", "right", "metric", "mean_difference", "sd_difference", "ci_low", "ci_high", "p_value", "positive_sign_count", "n", "holm_p_value", "ci_gate", "holm_gate", "sign_gate", "gate_pass"])
    write_csv(output_dir / "paired_differences.csv", difference_rows, ["seed", "contrast", "metric", "difference"])
    aggregate_payload["config"] = {key: (list(value) if isinstance(value, tuple) else value) for key, value in vars(cfg).items()}
    aggregate_payload["spillover"] = {"mean_absolute_q_X_difference": float(np.mean(spillovers)), "minimum": float(np.min(spillovers)), "maximum": float(np.max(spillovers))}
    write_json(output_dir / "summary.json", aggregate_payload)
    make_figure(curve_rows, output_dir)
    write_table(output_dir / "table_snippet.tex", summary_rows)
    return aggregate_payload


def main() -> int:
    args = parse_args()
    base = Path(__file__).resolve().parent
    protocol = base / "PROTOCOL.md"
    protocol_hash = sha256(protocol)
    formal = bool(args.formal)
    if formal and protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit(f"protocol hash mismatch: expected {EXPECTED_PROTOCOL_SHA256}, found {protocol_hash}")
    cfg = Config() if formal else replace(Config(), seeds=(-2, -1), updates=10, record_every=5)
    output_dir = args.output_dir or base / ("results_formal" if formal else "engineering_smoke")
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    started = time.monotonic()
    summary = execute(cfg, output_dir, formal)
    replay: dict[str, object] = {"requested": bool(args.verify_determinism), "pass": None, "matched": 0, "total": len(CORE_ARTIFACTS)}
    if args.verify_determinism:
        with tempfile.TemporaryDirectory(prefix="ace_shared_replay_") as temporary:
            replay_dir = Path(temporary) / "replay"
            execute(cfg, replay_dir, formal)
            matches = {name: sha256(output_dir / name) == sha256(replay_dir / name) for name in CORE_ARTIFACTS}
            replay = {"requested": True, "pass": bool(all(matches.values())), "matched": int(sum(matches.values())), "total": len(matches), "artifacts": matches}
            if not replay["pass"]:
                raise RuntimeError(f"determinism replay failed: {matches}")
    duration = time.monotonic() - started
    write_json(output_dir / "determinism_receipt.json", replay)
    manifest = {
        "experiment": "shared_representation_selectivity_v1", "formal": formal,
        "command": " ".join(shlex.quote(part) for part in sys.argv), "duration_seconds": duration,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "protocol_sha256": protocol_hash, "code_sha256": sha256(Path(__file__).resolve()),
        "seed_count": len(cfg.seeds), "method_count": len(METHODS), "failure_count": sum(1 for _ in csv.DictReader((output_dir / "failures.csv").open(encoding="utf-8"))),
        "scientific_decision": summary["summary"]["scientific_decision"], "determinism": replay,
    }
    write_json(output_dir / "manifest.json", manifest)
    artifact_names = sorted(path.name for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output_dir / "checksums.sha256").write_text("".join(f"{sha256(output_dir / name)}  {name}\n" for name in artifact_names), encoding="utf-8")
    if not args.quiet:
        print(json.dumps({"output_dir": str(output_dir), "duration_seconds": duration, "decision": manifest["scientific_decision"], "failures": manifest["failure_count"], "determinism": replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
