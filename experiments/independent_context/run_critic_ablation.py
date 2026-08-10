#!/usr/bin/env python3
"""Prospectively frozen randomized critic-isolation ablation (A3)."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPECTED_PROTOCOL_SHA256 = (
    "18611fef078664e7633e24f6a436c08df12492df5ea9f090532f200eade3f5ed"
)
VARIANTS = ("full_disjoint", "shared_encoder_no_stop")
DISPLAY_NAMES = {
    "full_disjoint": "Full disjoint",
    "shared_encoder_no_stop": "Shared encoder, no stop-gradient",
}
CONTEXT_NAMES = ("V", "I", "P")
PARAMETER_NAMES = (
    "encoder.weight",
    "encoder.bias",
    "value_head.weight",
    "value_head.bias",
)
FORMAL_SEEDS = tuple(range(2000, 2100))
SMOKE_SEEDS = (-2, -1)
T_CRITICAL_95 = {
    1: 12.706204736432095,
    99: 1.9842169515086827,
}


@dataclass(frozen=True)
class Config:
    hidden_width: int = 8
    contexts: int = 3
    actions: int = 3
    batch_size: int = 64
    response_action_indices: tuple[int, int] = (1, 2)
    audit_coefficient: float = 1.0
    learning_rate: float = 0.05
    full_zero_tolerance: float = 1e-12
    shared_encoder_delta_threshold: float = 1e-10
    shared_output_change_threshold: float = 1e-8
    shared_output_required_count: int = 95


@dataclass(frozen=True)
class TensorBank:
    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor
    actor_head_weight: torch.Tensor
    actor_head_bias: torch.Tensor
    value_head_weight: torch.Tensor
    value_head_bias: torch.Tensor
    contexts: torch.Tensor


class DisjointActorCritic(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.actor_encoder = nn.Linear(cfg.contexts, cfg.hidden_width)
        self.critic_encoder = nn.Linear(cfg.contexts, cfg.hidden_width)
        self.actor_head = nn.Linear(cfg.hidden_width, cfg.actions)
        self.value_head = nn.Linear(cfg.hidden_width, 1)

    def actor_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_head(torch.tanh(self.actor_encoder(observations)))

    def critic_output(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.tanh(self.critic_encoder(observations))).squeeze(-1)


class SharedActorCritic(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = nn.Linear(cfg.contexts, cfg.hidden_width)
        self.actor_head = nn.Linear(cfg.hidden_width, cfg.actions)
        self.value_head = nn.Linear(cfg.hidden_width, 1)

    def actor_logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_head(torch.tanh(self.encoder(observations)))

    def critic_output(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.tanh(self.encoder(observations))).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Artifact directory. Defaults to results_critic for the formal run and "
            "smoke_critic for --engineering-smoke."
        ),
    )
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="Use only frozen smoke seeds -2 and -1; never evaluate the formal gate.",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Replay once in an isolated temporary directory and compare core artifacts.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal summary.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    if rows:
        fieldnames = list(rows[0]) if fieldnames is None else fieldnames
    elif fieldnames is None:
        raise ValueError(f"fieldnames are required for an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seeded_uniform(
    generator: torch.Generator, shape: tuple[int, ...], fan_in: int
) -> torch.Tensor:
    """Match nn.Linear.reset_parameters' uniform bound in deterministic float64."""
    bound = 1.0 / math.sqrt(fan_in)
    values = torch.rand(shape, generator=generator, dtype=torch.float64)
    return values.mul(2.0 * bound).sub(bound)


def make_tensor_bank(seed: int, cfg: Config) -> TensorBank:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    encoder_weight = seeded_uniform(
        generator, (cfg.hidden_width, cfg.contexts), cfg.contexts
    )
    encoder_bias = seeded_uniform(generator, (cfg.hidden_width,), cfg.contexts)
    actor_head_weight = seeded_uniform(
        generator, (cfg.actions, cfg.hidden_width), cfg.hidden_width
    )
    actor_head_bias = seeded_uniform(generator, (cfg.actions,), cfg.hidden_width)
    value_head_weight = seeded_uniform(
        generator, (1, cfg.hidden_width), cfg.hidden_width
    )
    value_head_bias = seeded_uniform(generator, (1,), cfg.hidden_width)
    contexts = torch.randint(
        low=0,
        high=cfg.contexts,
        size=(cfg.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    return TensorBank(
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        actor_head_weight=actor_head_weight,
        actor_head_bias=actor_head_bias,
        value_head_weight=value_head_weight,
        value_head_bias=value_head_bias,
        contexts=contexts,
    )


def copy_linear(linear: nn.Linear, weight: torch.Tensor, bias: torch.Tensor) -> None:
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)


def build_model(
    variant: str, bank: TensorBank, cfg: Config
) -> tuple[nn.Module, list[nn.Parameter], dict[str, nn.Parameter]]:
    if variant == "full_disjoint":
        model = DisjointActorCritic(cfg).to(dtype=torch.float64)
        copy_linear(model.actor_encoder, bank.encoder_weight, bank.encoder_bias)
        copy_linear(model.critic_encoder, bank.encoder_weight, bank.encoder_bias)
        copy_linear(model.actor_head, bank.actor_head_weight, bank.actor_head_bias)
        copy_linear(model.value_head, bank.value_head_weight, bank.value_head_bias)
        stepped_parameters = list(model.actor_encoder.parameters()) + list(
            model.actor_head.parameters()
        )
        critic_used = {
            "encoder.weight": model.critic_encoder.weight,
            "encoder.bias": model.critic_encoder.bias,
            "value_head.weight": model.value_head.weight,
            "value_head.bias": model.value_head.bias,
        }
    elif variant == "shared_encoder_no_stop":
        model = SharedActorCritic(cfg).to(dtype=torch.float64)
        copy_linear(model.encoder, bank.encoder_weight, bank.encoder_bias)
        copy_linear(model.actor_head, bank.actor_head_weight, bank.actor_head_bias)
        copy_linear(model.value_head, bank.value_head_weight, bank.value_head_bias)
        stepped_parameters = list(model.encoder.parameters()) + list(
            model.actor_head.parameters()
        )
        critic_used = {
            "encoder.weight": model.encoder.weight,
            "encoder.bias": model.encoder.bias,
            "value_head.weight": model.value_head.weight,
            "value_head.bias": model.value_head.bias,
        }
    else:  # pragma: no cover - guarded by the fixed variant list
        raise ValueError(f"unknown variant: {variant}")
    return model, stepped_parameters, critic_used


def response_probability(logits: torch.Tensor, cfg: Config) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    return probabilities[:, list(cfg.response_action_indices)].sum(dim=-1)


def tensor_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach()).item())


def aggregate_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def run_variant(
    seed: int, variant: str, bank: TensorBank, cfg: Config
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, stepped_parameters, critic_used = build_model(variant, bank, cfg)
    all_contexts = torch.eye(cfg.contexts, dtype=torch.float64)
    audit_observations = all_contexts[bank.contexts]
    before_parameters = {
        name: parameter.detach().clone() for name, parameter in critic_used.items()
    }
    actor_head_before = {
        name: parameter.detach().clone()
        for name, parameter in model.actor_head.named_parameters()
    }

    with torch.no_grad():
        batch_response_before = response_probability(
            model.actor_logits(audit_observations), cfg
        )
        context_response_before = response_probability(
            model.actor_logits(all_contexts), cfg
        )
        critic_before = model.critic_output(all_contexts)

    loss = -cfg.audit_coefficient * torch.log(
        response_probability(model.actor_logits(audit_observations), cfg)
    ).mean()
    optimizer = torch.optim.SGD(stepped_parameters, lr=cfg.learning_rate)
    model.zero_grad(set_to_none=True)
    loss.backward()

    gradient_norms = {
        name: tensor_norm(parameter.grad) for name, parameter in critic_used.items()
    }
    optimizer.step()

    parameter_delta_norms = {
        name: tensor_norm(parameter.detach() - before_parameters[name])
        for name, parameter in critic_used.items()
    }
    actor_head_delta_norms = {
        name: tensor_norm(parameter.detach() - actor_head_before[name])
        for name, parameter in model.actor_head.named_parameters()
    }
    with torch.no_grad():
        batch_response_after = response_probability(
            model.actor_logits(audit_observations), cfg
        )
        context_response_after = response_probability(
            model.actor_logits(all_contexts), cfg
        )
        critic_after = model.critic_output(all_contexts)

    response_before = float(batch_response_before.mean().item())
    response_after = float(batch_response_after.mean().item())
    critic_output_change = float((critic_after - critic_before).abs().max().item())
    critic_gradient_norm = aggregate_norm(gradient_norms.values())
    critic_parameter_delta = aggregate_norm(parameter_delta_norms.values())
    encoder_delta = aggregate_norm(
        parameter_delta_norms[name]
        for name in ("encoder.weight", "encoder.bias")
    )
    value_head_delta = aggregate_norm(
        parameter_delta_norms[name]
        for name in ("value_head.weight", "value_head.bias")
    )
    row: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "batch_size": cfg.batch_size,
        "batch_count_V": int((bank.contexts == 0).sum().item()),
        "batch_count_I": int((bank.contexts == 1).sum().item()),
        "batch_count_P": int((bank.contexts == 2).sum().item()),
        "audit_loss_before": float(loss.detach().item()),
        "response_probability_before": response_before,
        "response_probability_after": response_after,
        "response_probability_delta": response_after - response_before,
        "critic_used_gradient_norm": critic_gradient_norm,
        "critic_used_parameter_delta_norm": critic_parameter_delta,
        "encoder_parameter_delta_norm": encoder_delta,
        "value_head_parameter_delta_norm": value_head_delta,
        "actor_head_parameter_delta_norm": aggregate_norm(
            actor_head_delta_norms.values()
        ),
        "critic_output_max_abs_change": critic_output_change,
    }
    for index, context_name in enumerate(CONTEXT_NAMES):
        row[f"response_{context_name}_before"] = float(
            context_response_before[index].item()
        )
        row[f"response_{context_name}_after"] = float(
            context_response_after[index].item()
        )
        row[f"critic_{context_name}_before"] = float(critic_before[index].item())
        row[f"critic_{context_name}_after"] = float(critic_after[index].item())

    parameter_rows = [
        {
            "seed": seed,
            "variant": variant,
            "critic_used_parameter": name,
            "audit_gradient_norm": gradient_norms[name],
            "parameter_delta_norm": parameter_delta_norms[name],
        }
        for name in PARAMETER_NAMES
    ]
    return row, parameter_rows


def mean_sd_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n not in (2, 100):
        raise ValueError(f"expected 2 smoke or 100 formal values, got {n}")
    mean = math.fsum(values) / n
    variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
    sd = math.sqrt(max(variance, 0.0))
    critical = T_CRITICAL_95[n - 1]
    half_width = critical * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "response_probability_before",
        "response_probability_after",
        "response_probability_delta",
        "critic_used_gradient_norm",
        "critic_used_parameter_delta_norm",
        "encoder_parameter_delta_norm",
        "value_head_parameter_delta_norm",
        "critic_output_max_abs_change",
    )
    summaries: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary: dict[str, Any] = {
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "n_seeds": len(selected),
        }
        for metric in metrics:
            stats = mean_sd_ci([float(row[metric]) for row in selected])
            for statistic, value in stats.items():
                summary[f"{metric}_{statistic}"] = value
        summaries.append(summary)
    return summaries


def paired_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "response_probability_delta",
        "critic_used_gradient_norm",
        "critic_used_parameter_delta_norm",
        "encoder_parameter_delta_norm",
        "value_head_parameter_delta_norm",
        "critic_output_max_abs_change",
    )
    indexed = {
        (int(row["seed"]), str(row["variant"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    contrasts: list[dict[str, Any]] = []
    for metric in metrics:
        differences = [
            float(indexed[(seed, "shared_encoder_no_stop")][metric])
            - float(indexed[(seed, "full_disjoint")][metric])
            for seed in seeds
        ]
        contrasts.append(
            {
                "contrast": "shared_encoder_no_stop-minus-full_disjoint",
                "metric": metric,
                **mean_sd_ci(differences),
            }
        )
    return contrasts


def validate(
    rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    seeds: tuple[int, ...],
    engineering_smoke: bool,
    cfg: Config,
) -> dict[str, Any]:
    expected_runs = len(seeds) * len(VARIANTS)
    structural_errors: list[str] = []
    if failures:
        structural_errors.append(f"failure ledger contains {len(failures)} rows")
    if len(rows) != expected_runs:
        structural_errors.append(f"completed {len(rows)} of {expected_runs} runs")
    if len(parameter_rows) != expected_runs * len(PARAMETER_NAMES):
        structural_errors.append(
            f"recorded {len(parameter_rows)} of "
            f"{expected_runs * len(PARAMETER_NAMES)} critic-used parameter rows"
        )
    keys = {
        (int(row["seed"]), str(row["variant"])) for row in rows
    }
    expected_keys = {(seed, variant) for seed in seeds for variant in VARIANTS}
    if keys != expected_keys:
        structural_errors.append("seed-variant keys are incomplete or duplicated")
    numeric_fields = (
        "response_probability_before",
        "response_probability_after",
        "response_probability_delta",
        "critic_used_gradient_norm",
        "critic_used_parameter_delta_norm",
        "encoder_parameter_delta_norm",
        "value_head_parameter_delta_norm",
        "critic_output_max_abs_change",
    )
    for row in rows:
        if not all(math.isfinite(float(row[field])) for field in numeric_fields):
            structural_errors.append(
                f"non-finite value for seed={row['seed']} variant={row['variant']}"
            )
        for field in ("response_probability_before", "response_probability_after"):
            if not 0.0 <= float(row[field]) <= 1.0:
                structural_errors.append(
                    f"invalid probability for seed={row['seed']} "
                    f"variant={row['variant']} field={field}"
                )

    full = [row for row in rows if row["variant"] == "full_disjoint"]
    shared = [row for row in rows if row["variant"] == "shared_encoder_no_stop"]
    counts = {
        "full_response_increase": sum(
            float(row["response_probability_delta"]) > 0.0 for row in full
        ),
        "full_critic_gradient_at_most_1e-12": sum(
            float(row["critic_used_gradient_norm"]) <= cfg.full_zero_tolerance
            for row in full
        ),
        "full_critic_parameter_delta_at_most_1e-12": sum(
            float(row["critic_used_parameter_delta_norm"])
            <= cfg.full_zero_tolerance
            for row in full
        ),
        "full_critic_output_change_at_most_1e-12": sum(
            float(row["critic_output_max_abs_change"]) <= cfg.full_zero_tolerance
            for row in full
        ),
        "shared_response_increase": sum(
            float(row["response_probability_delta"]) > 0.0 for row in shared
        ),
        "shared_encoder_delta_above_1e-10": sum(
            float(row["encoder_parameter_delta_norm"])
            > cfg.shared_encoder_delta_threshold
            for row in shared
        ),
        "shared_value_head_delta_at_most_1e-12": sum(
            float(row["value_head_parameter_delta_norm"]) <= cfg.full_zero_tolerance
            for row in shared
        ),
        "shared_critic_output_change_above_1e-8": sum(
            float(row["critic_output_max_abs_change"])
            > cfg.shared_output_change_threshold
            for row in shared
        ),
    }
    if engineering_smoke:
        smoke_checks = {
            key: count == len(seeds) for key, count in counts.items()
        }
        return {
            "run_mode": "engineering_smoke",
            "structural_pass": not structural_errors,
            "structural_errors": structural_errors,
            "counts": counts,
            "engineering_smoke_checks": smoke_checks,
            "engineering_smoke_status": (
                "PASS" if not structural_errors and all(smoke_checks.values()) else "FAIL"
            ),
            "formal_structural_gate": "NOT_EVALUATED_ENGINEERING_SMOKE",
            "claim_boundary": (
                "Smoke seeds -2 and -1 test code structure only and cannot support "
                "the frozen 100-seed A3 claim."
            ),
        }

    conditions = {
        "full_response_increase_100_of_100": counts["full_response_increase"] == 100,
        "full_critic_gradient_zero_100_of_100": counts[
            "full_critic_gradient_at_most_1e-12"
        ]
        == 100,
        "full_critic_parameter_delta_zero_100_of_100": counts[
            "full_critic_parameter_delta_at_most_1e-12"
        ]
        == 100,
        "full_critic_output_change_zero_100_of_100": counts[
            "full_critic_output_change_at_most_1e-12"
        ]
        == 100,
        "shared_response_increase_100_of_100": counts["shared_response_increase"]
        == 100,
        "shared_encoder_delta_positive_100_of_100": counts[
            "shared_encoder_delta_above_1e-10"
        ]
        == 100,
        "shared_value_head_delta_zero_100_of_100": counts[
            "shared_value_head_delta_at_most_1e-12"
        ]
        == 100,
        "shared_critic_output_change_at_least_95_of_100": counts[
            "shared_critic_output_change_above_1e-8"
        ]
        >= cfg.shared_output_required_count,
    }
    gate_pass = not structural_errors and all(conditions.values())
    return {
        "run_mode": "formal",
        "structural_pass": not structural_errors,
        "structural_errors": structural_errors,
        "counts": counts,
        "formal_conditions": conditions,
        "formal_structural_gate": "PASS" if gate_pass else "FAIL",
        "claim_boundary": (
            "A3 tests audit-to-critic parameter leakage structurally; it is not "
            "financial-performance or market-effect evidence."
        ),
    }


def format_ci(row: dict[str, Any], metric: str, scientific: bool = False) -> str:
    formatter = ".3e" if scientific else ".5f"
    mean = format(float(row[f"{metric}_mean"]), formatter)
    low = format(float(row[f"{metric}_ci95_low"]), formatter)
    high = format(float(row[f"{metric}_ci95_high"]), formatter)
    return f"{mean} [{low}, {high}]"


def write_tex_table(
    summaries: list[dict[str, Any]], validation: dict[str, Any], output_dir: Path
) -> None:
    run_mode = str(validation["run_mode"])
    n = int(summaries[0]["n_seeds"])
    if run_mode == "engineering_smoke":
        caption = (
            "Engineering smoke for the randomized critic-isolation ablation "
            "($n=2$, seeds $-2,-1$). Both routes increase response-set mass. "
            "Only the shared encoder changes the critic output; the frozen "
            "100-seed gate is not evaluated. Values are means with 95\\% "
            "paired-seed $t$ intervals."
        )
    else:
        caption = (
            "Randomized critic-isolation ablation over 100 matched-initialization "
            "seeds. The disjoint audit route leaves critic-used parameters and "
            "outputs unchanged, whereas the no-stop shared encoder changes both "
            "the shared representation and critic output. Values are means with "
            "95\\% paired-seed $t$ intervals."
        )
    lines = [
        "% Auto-generated by run_critic_ablation.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:critic_isolation_ablation}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & $\\Delta P(\\{M,A\\})$ & Critic-used parameter $\\Delta$ & Value-head $\\Delta$ & Max critic-output $\\Delta$ \\\\",
        "\\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{row['display_name']} & "
            f"{format_ci(row, 'response_probability_delta')} & "
            f"{format_ci(row, 'critic_used_parameter_delta_norm', scientific=True)} & "
            f"{format_ci(row, 'value_head_parameter_delta_norm', scientific=True)} & "
            f"{format_ci(row, 'critic_output_max_abs_change', scientific=True)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (output_dir / "table_critic_ablation.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_readme(output_dir: Path, validation: dict[str, Any]) -> None:
    mode_note = (
        "This directory is an engineering smoke receipt only; it cannot be cited "
        "as the formal A3 result."
        if validation["run_mode"] == "engineering_smoke"
        else "This directory contains the frozen formal A3 run."
    )
    lines = [
        "# Randomized critic-isolation ablation artifacts",
        "",
        mode_note,
        "",
        "The ablation uses matched float64 initialization and the same 64-row "
        "audit batch for both variants within each seed. The response set is "
        "`{M,A}`. It tests gradient routing only, not financial or market performance.",
        "",
        "## Frozen formal command",
        "",
        "```bash",
        "python3 run_critic_ablation.py --output-dir results_critic --verify-determinism",
        "```",
        "",
        "## Artifact map",
        "",
        "- `seed_results.csv`: one raw row per seed and variant.",
        "- `parameter_traces.csv`: gradient and parameter-delta norms for every critic-used parameter.",
        "- `summary.csv` and `summary.json`: variant summaries, paired contrasts, and the gate receipt.",
        "- `paired_contrasts.csv`: shared-minus-disjoint paired differences.",
        "- `failures.csv`: header-preserving failure ledger.",
        "- `table_critic_ablation.tex`: include-ready table.",
        "- `manifest.json` and `checksums.sha256`: execution and provenance receipt.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest_and_checksums(
    output_dir: Path,
    source_dir: Path,
    args: argparse.Namespace,
    cfg: Config,
    seeds: tuple[int, ...],
    validation: dict[str, Any],
    completed_runs: int,
    failure_count: int,
    duration_seconds: float,
    artifact_names: list[str],
    replay: dict[str, Any] | None = None,
) -> None:
    artifact_hashes = {name: sha256(output_dir / name) for name in artifact_names}
    command = " ".join(shlex.quote(part) for part in sys.argv)
    manifest = {
        "schema_version": 1,
        "status": "complete" if validation["structural_pass"] else "failed",
        "run_mode": validation["run_mode"],
        "formal_structural_gate": validation["formal_structural_gate"],
        "claim_boundary": validation["claim_boundary"],
        "command": command,
        "duration_seconds": round(duration_seconds, 6),
        "protocol_status": "frozen before implementation and first run",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "script_sha256": sha256(source_dir / "run_critic_ablation.py"),
        "config": asdict(cfg),
        "seeds": list(seeds),
        "initialization": (
            "matched PyTorch nn.Linear default-bound uniform tensors in float64; "
            "the disjoint actor and critic encoders start from identical copies"
        ),
        "audit_batch": (
            "64 contexts sampled uniformly from {V,I,P} with the per-seed torch "
            "Generator and reused exactly across variants"
        ),
        "expected_seed_variant_runs": len(seeds) * len(VARIANTS),
        "completed_seed_variant_runs": completed_runs,
        "failure_count": failure_count,
        "validation": validation,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "artifact_sha256": artifact_hashes,
        "deterministic_replay": replay,
    }
    write_json(output_dir / "manifest.json", manifest)
    ledger = dict(artifact_hashes)
    ledger["manifest.json"] = sha256(output_dir / "manifest.json")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items())),
        encoding="utf-8",
    )


def verify_determinism(
    output_dir: Path,
    source_dir: Path,
    engineering_smoke: bool,
) -> dict[str, Any]:
    core_artifacts = [
        "failures.csv",
        "paired_contrasts.csv",
        "parameter_traces.csv",
        "seed_results.csv",
        "summary.csv",
        "summary.json",
        "table_critic_ablation.tex",
    ]
    replay_dir = Path(tempfile.mkdtemp(prefix="ace_critic_isolation_replay_"))
    try:
        command = [
            sys.executable,
            str(source_dir / "run_critic_ablation.py"),
            "--output-dir",
            str(replay_dir),
            "--quiet",
        ]
        if engineering_smoke:
            command.append("--engineering-smoke")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "isolated deterministic replay failed: "
                f"returncode={completed.returncode}; stderr={completed.stderr.strip()}"
            )
        mismatches = [
            name
            for name in core_artifacts
            if (output_dir / name).read_bytes() != (replay_dir / name).read_bytes()
        ]
        receipt = {
            "status": "PASS" if not mismatches else "FAIL",
            "comparison": "byte-for-byte isolated replay of core scientific artifacts",
            "compared_artifact_count": len(core_artifacts),
            "matched_artifact_count": len(core_artifacts) - len(mismatches),
            "mismatched_artifacts": mismatches,
            "comparison_sha256": {
                name: sha256(output_dir / name) for name in core_artifacts
            },
        }
        write_json(output_dir / "determinism_receipt.json", receipt)
        if mismatches:
            raise RuntimeError(
                "isolated replay mismatched: " + ", ".join(mismatches)
            )
        return receipt
    finally:
        shutil.rmtree(replay_dir)


def main() -> int:
    args = parse_args()
    source_dir = Path(__file__).resolve().parent
    protocol_path = source_dir / "PROTOCOL.md"
    actual_protocol_hash = sha256(protocol_path)
    if actual_protocol_hash != EXPECTED_PROTOCOL_SHA256:
        print(
            "ERROR: PROTOCOL.md hash changed; refusing to run. "
            f"expected={EXPECTED_PROTOCOL_SHA256} actual={actual_protocol_hash}",
            file=sys.stderr,
        )
        return 2

    cfg = Config()
    engineering_smoke = bool(args.engineering_smoke)
    seeds = SMOKE_SEEDS if engineering_smoke else FORMAL_SEEDS
    default_name = "smoke_critic" if engineering_smoke else "results_critic"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source_dir / default_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for seed in seeds:
        try:
            bank = make_tensor_bank(seed, cfg)
            for variant in VARIANTS:
                row, traces = run_variant(seed, variant, bank, cfg)
                rows.append(row)
                parameter_rows.extend(traces)
        except Exception as exc:  # pragma: no cover - retained as explicit ledger
            failures.append(
                {
                    "seed": seed,
                    "variant": "seed_pair",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    write_csv(output_dir / "seed_results.csv", rows)
    write_csv(output_dir / "parameter_traces.csv", parameter_rows)
    write_csv(
        output_dir / "failures.csv",
        failures,
        fieldnames=["seed", "variant", "exception_type", "message"],
    )

    expected_runs = len(seeds) * len(VARIANTS)
    if failures or len(rows) != expected_runs:
        print(
            "ERROR: incomplete A3 run; raw rows and failure ledger were preserved.",
            file=sys.stderr,
        )
        return 2

    summaries = summarize_variants(rows)
    contrasts = paired_contrasts(rows)
    validation = validate(
        rows, parameter_rows, failures, seeds, engineering_smoke, cfg
    )
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "paired_contrasts.csv", contrasts)
    write_json(
        output_dir / "summary.json",
        {
            "run_mode": validation["run_mode"],
            "claim_boundary": validation["claim_boundary"],
            "config": asdict(cfg),
            "seeds": list(seeds),
            "variant_summaries": summaries,
            "paired_contrasts": contrasts,
            "validation": validation,
        },
    )
    write_tex_table(summaries, validation, output_dir)
    write_readme(output_dir, validation)

    deterministic_replay = None
    if args.verify_determinism:
        deterministic_replay = verify_determinism(
            output_dir, source_dir, engineering_smoke
        )
    duration_seconds = time.perf_counter() - start
    artifact_names = [
        "README.md",
        "failures.csv",
        "paired_contrasts.csv",
        "parameter_traces.csv",
        "seed_results.csv",
        "summary.csv",
        "summary.json",
        "table_critic_ablation.tex",
    ]
    if deterministic_replay is not None:
        artifact_names.append("determinism_receipt.json")
    write_manifest_and_checksums(
        output_dir=output_dir,
        source_dir=source_dir,
        args=args,
        cfg=cfg,
        seeds=seeds,
        validation=validation,
        completed_runs=len(rows),
        failure_count=len(failures),
        duration_seconds=duration_seconds,
        artifact_names=artifact_names,
        replay=deterministic_replay,
    )

    if not args.quiet:
        full = next(row for row in summaries if row["variant"] == "full_disjoint")
        shared = next(
            row for row in summaries if row["variant"] == "shared_encoder_no_stop"
        )
        print(f"Run mode: {validation['run_mode']}")
        print(f"Completed seed-variant runs: {len(rows)}/{expected_runs}")
        print(
            "Mean response-probability change: "
            f"full={full['response_probability_delta_mean']:.12g}, "
            f"shared={shared['response_probability_delta_mean']:.12g}"
        )
        print(
            "Mean max critic-output change: "
            f"full={full['critic_output_max_abs_change_mean']:.12g}, "
            f"shared={shared['critic_output_max_abs_change_mean']:.12g}"
        )
        if engineering_smoke:
            print(
                "Engineering smoke: "
                f"{validation['engineering_smoke_status']}; formal gate NOT EVALUATED"
            )
        else:
            print(f"Formal structural gate: {validation['formal_structural_gate']}")
        if deterministic_replay is not None:
            print(
                "Deterministic replay: "
                f"{deterministic_replay['status']} "
                f"({deterministic_replay['matched_artifact_count']}/"
                f"{deterministic_replay['compared_artifact_count']} byte-identical)"
            )
        print(f"Artifacts: {output_dir}")
    return 0 if validation["structural_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
