#!/usr/bin/env python3
"""Prospectively frozen audit-coefficient sensitivity assay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shlex
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


EXPECTED_PROTOCOL_SHA256 = (
    "cd0cb764f2fac84b6c011428114981039bdc3d9380a3fc7b8ad3707045791b56"
)
ALPHAS = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
FORMAL_SEEDS = tuple(range(6000, 6050))
CORE_ARTIFACTS = (
    "seed_results.csv",
    "summary.csv",
    "summary.json",
    "paired_checks.csv",
    "failures.csv",
    "learning_curves.csv",
    "RESULTS.md",
)


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


def load_main_module(source_dir: Path):
    path = source_dir.parent / "independent_context" / "run_main_experiment.py"
    spec = importlib.util.spec_from_file_location("ace_main_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen main runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size < 2:
        raise ValueError("at least two completed seeds are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, values.size - 1) * sd / math.sqrt(values.size))
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def execute_core(output_dir: Path, formal: bool, module) -> dict:
    seeds = FORMAL_SEEDS if formal else (-2, -1)
    updates = 400 if formal else 20
    batch_size = 384 if formal else 64
    base_cfg = module.Config(
        seeds=tuple(seeds),
        updates=updates,
        batch_size=batch_size,
        learning_rate=0.08,
        initial_logit_sd=0.15,
        reward_noise_sd=0.10,
        actor_alpha=0.0,
        record_every=10,
    )
    seed_rows: list[dict] = []
    curve_rows: list[dict] = []
    failures: list[dict] = []
    for seed in seeds:
        stream = module.make_seed_stream(seed, base_cfg)
        for alpha in ALPHAS:
            cfg = replace(base_cfg, actor_alpha=alpha)
            try:
                trajectories, _diagnostics, final = module.train_method(
                    seed, "full_ace", stream, cfg
                )
                seed_rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "q_V": final["q_V"],
                        "q_X": final["q_X"],
                        "selectivity_gap": final["selectivity_gap"],
                        "U": final["U"],
                        "maximum_probability_sum_error": final[
                            "maximum_probability_sum_error"
                        ],
                        "finite_run": final["finite_run"],
                    }
                )
                for row in trajectories:
                    curve_rows.append(
                        {
                            "seed": seed,
                            "alpha": alpha,
                            "update": row["update"],
                            "q_V": row["q_V"],
                            "q_X": row["q_X"],
                            "selectivity_gap": row["selectivity_gap"],
                            "U": row["U"],
                        }
                    )
            except Exception as exc:  # retain every failed planned row
                failures.append(
                    {"seed": seed, "alpha": alpha, "error_type": type(exc).__name__, "message": str(exc)}
                )

    summary_rows: list[dict] = []
    metrics = ("q_V", "q_X", "selectivity_gap", "U")
    for alpha in ALPHAS:
        rows = [row for row in seed_rows if float(row["alpha"]) == alpha]
        summary_row: dict[str, float | int] = {"alpha": alpha, "n_seeds": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            for statistic, value in summarize(values).items():
                summary_row[f"{metric}_{statistic}"] = value
        summary_rows.append(summary_row)

    index = {(int(row["seed"]), float(row["alpha"])): row for row in seed_rows}
    check_rows: list[dict] = []
    for alpha in ALPHAS[1:]:
        differences = np.asarray(
            [
                float(index[(seed, alpha)]["q_V"]) - float(index[(seed, 0.0)]["q_V"])
                for seed in seeds
            ],
            dtype=np.float64,
        )
        qx_differences = np.asarray(
            [
                abs(float(index[(seed, alpha)]["q_X"]) - float(index[(seed, 0.0)]["q_X"]))
                for seed in seeds
            ],
            dtype=np.float64,
        )
        statistics = summarize(differences)
        sign_count = int(np.sum(differences > 0.0))
        direction_pass = bool(
            formal and statistics["ci95_low"] > 0.0 and sign_count >= 45
        )
        isolation_pass = bool(float(qx_differences.max()) <= 1e-12)
        check_rows.append(
            {
                "alpha": alpha,
                "n_pairs": len(differences),
                "q_V_difference_mean": statistics["mean"],
                "q_V_difference_ci95_low": statistics["ci95_low"],
                "q_V_difference_ci95_high": statistics["ci95_high"],
                "positive_pairs": sign_count,
                "direction_gate_pass": int(direction_pass) if formal else "",
                "maximum_absolute_q_X_difference": float(qx_differences.max()),
                "invalid_context_isolation_pass": int(isolation_pass),
            }
        )

    qv_means = [float(row["q_V_mean"]) for row in summary_rows]
    dose_ordering = all(right >= left for left, right in zip(qv_means, qv_means[1:]))
    complete = len(seed_rows) == len(seeds) * len(ALPHAS) and not failures
    finite = all(int(row["finite_run"]) == 1 for row in seed_rows)
    valid_probabilities = all(
        float(row["maximum_probability_sum_error"]) <= 1e-12 for row in seed_rows
    )
    direction_all = formal and all(int(row["direction_gate_pass"]) == 1 for row in check_rows)
    isolation_all = all(int(row["invalid_context_isolation_pass"]) == 1 for row in check_rows)
    decision = (
        "GO" if complete and finite and valid_probabilities and direction_all and isolation_all else "NO-GO"
    ) if formal else "ENGINEERING-SMOKE-ONLY"
    validation = {
        "run_mode": "formal" if formal else "engineering_smoke",
        "expected_runs": len(seeds) * len(ALPHAS),
        "completed_runs": len(seed_rows),
        "failure_count": len(failures),
        "complete": complete,
        "finite": finite,
        "valid_probabilities": valid_probabilities,
        "all_direction_gates_pass": bool(direction_all),
        "all_invalid_context_isolation_checks_pass": isolation_all,
        "q_V_mean_nondecreasing_over_grid": dose_ordering,
        "scientific_decision": decision,
        "claim_permitted": bool(formal and decision == "GO"),
    }

    write_csv(
        output_dir / "seed_results.csv",
        seed_rows,
        ["seed", "alpha", "q_V", "q_X", "selectivity_gap", "U", "maximum_probability_sum_error", "finite_run"],
    )
    write_csv(output_dir / "summary.csv", summary_rows, list(summary_rows[0]))
    write_json(output_dir / "summary.json", {"config": asdict(base_cfg), "alphas": list(ALPHAS), "rows": summary_rows, "validation": validation})
    write_csv(output_dir / "paired_checks.csv", check_rows, list(check_rows[0]))
    write_csv(output_dir / "failures.csv", failures, ["seed", "alpha", "error_type", "message"])
    write_csv(output_dir / "learning_curves.csv", curve_rows, list(curve_rows[0]))

    lines = [
        "# ACE-RL audit-coefficient sensitivity",
        "",
        f"Mode: **{validation['run_mode']}**. Decision: **{decision}**.",
        "",
        "| alpha | q_V mean [95% CI] | q_X mean [95% CI] | Selectivity mean [95% CI] | U mean [95% CI] |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['alpha']:.2f} | {row['q_V_mean']:.3f} [{row['q_V_ci95_low']:.3f}, {row['q_V_ci95_high']:.3f}] | "
            f"{row['q_X_mean']:.3f} [{row['q_X_ci95_low']:.3f}, {row['q_X_ci95_high']:.3f}] | "
            f"{row['selectivity_gap_mean']:.3f} [{row['selectivity_gap_ci95_low']:.3f}, {row['selectivity_gap_ci95_high']:.3f}] | "
            f"{row['U_mean']:.3f} [{row['U_ci95_low']:.3f}, {row['U_ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            f"- Complete runs: {validation['completed_runs']}/{validation['expected_runs']}; failures: {validation['failure_count']}.",
            f"- Every positive-alpha direction gate passes: {validation['all_direction_gates_pass']}.",
            f"- Every invalid-context isolation check passes: {validation['all_invalid_context_isolation_checks_pass']}.",
            f"- Mean q_V is nondecreasing over the frozen grid: {validation['q_V_mean_nondecreasing_over_grid']}.",
            "",
            "This study reports the response-strength and fixed-utility trade-off in the controlled independent-logit learner. It is not a long-run order-book profitability or collusion-mitigation result.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return {"config": base_cfg, "validation": validation}


def run_replay(output_dir: Path, formal: bool, module) -> dict:
    with tempfile.TemporaryDirectory(prefix="ace_alpha_replay_") as directory:
        replay_dir = Path(directory)
        execute_core(replay_dir, formal, module)
        mismatches = [
            name for name in CORE_ARTIFACTS if sha256(output_dir / name) != sha256(replay_dir / name)
        ]
    receipt = {"matched": not mismatches, "compared_artifacts": list(CORE_ARTIFACTS), "mismatches": mismatches}
    write_json(output_dir / "determinism_receipt.json", receipt)
    return receipt


def main() -> int:
    args = parse_args()
    source_dir = Path(__file__).resolve().parent
    protocol = source_dir / "PROTOCOL.md"
    if sha256(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit("frozen protocol hash mismatch")
    module, main_source = load_main_module(source_dir)
    formal = bool(args.formal)
    default_name = "results_formal" if formal else "engineering_smoke"
    output_dir = (args.output_dir or source_dir / default_name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = execute_core(output_dir, formal, module)
    replay = run_replay(output_dir, formal, module) if args.verify_determinism else None
    hashes = {name: sha256(output_dir / name) for name in CORE_ARTIFACTS}
    manifest = {
        "status": "complete" if result["validation"]["complete"] else "failed",
        "protocol_status": "frozen before implementation and formal result inspection",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "imported_main_runner_sha256": sha256(main_source),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "duration_seconds": time.perf_counter() - started,
        "alphas": list(ALPHAS),
        "config": asdict(result["config"]),
        "validation": result["validation"],
        "deterministic_replay": replay,
        "software": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "sha256": hashes,
    }
    write_json(output_dir / "manifest.json", manifest)
    checksum_hashes = {**hashes, "manifest.json": sha256(output_dir / "manifest.json")}
    if replay is not None:
        checksum_hashes["determinism_receipt.json"] = sha256(output_dir / "determinism_receipt.json")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksum_hashes.items())), encoding="utf-8"
    )
    if not args.quiet:
        print(json.dumps({"output_dir": str(output_dir), "validation": result["validation"], "deterministic_replay": replay}, indent=2, sort_keys=True))
    if formal and result["validation"]["scientific_decision"] != "GO":
        return 2
    if replay is not None and not replay["matched"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
