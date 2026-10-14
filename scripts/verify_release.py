#!/usr/bin/env python3
"""Fail-closed structural and integrity checks for this repository."""

from __future__ import annotations

import re
import json
import hashlib
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitignore",
    "README.md",
    "README.zh-CN.md",
    "requirements.txt",
    "ace_rl_project/__init__.py",
    "ace_rl_project/envs/__init__.py",
    "ace_rl_project/envs/lob_env.py",
    "ace_rl_project/experiments/__init__.py",
    "ace_rl_project/experiments/config.py",
    "assets/challenges.png",
    "assets/controlled.png",
    "assets/framework.png",
    "assets/logo-dark.png",
    "assets/logo.png",
    "assets/robustness.png",
    "assets/tendealer.png",
    "experiments/independent_context/PROTOCOL.md",
    "experiments/independent_context/run_main_experiment.py",
    "experiments/independent_context/run_critic_ablation.py",
    "experiments/independent_context/test_independent_context.py",
    "experiments/shared_representation/PROTOCOL.md",
    "experiments/shared_representation/run_experiment.py",
    "experiments/shared_representation/test_shared_representation.py",
    "experiments/coefficient_sensitivity/PROTOCOL.md",
    "experiments/coefficient_sensitivity/run_experiment.py",
    "experiments/coefficient_sensitivity/test_coefficient_sensitivity.py",
    "experiments/lob_contract/PROTOCOL.md",
    "experiments/lob_contract/run_experiment.py",
    "experiments/lob_contract/test_lob_contract.py",
    "paper/data/fig4/panel_a.csv",
    "paper/data/fig4/panel_b.csv",
    "paper/data/fig4/panel_c.csv",
    "paper/data/robustness.json",
    "paper/data/training_history.csv",
    "paper/extract_training_history.py",
    "paper/fig4_data_sd.py",
    "paper/plot_fig4.py",
    "paper/plot_robustness.py",
    "paper/plot_tendealer.py",
    "paper/robustness.py",
    "paper/table_stats.py",
    "revision/configs/abides_typed_audit_ippo_v2.json",
    "revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/rows.json",
    "revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/run_state.json",
    "revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/summary.json",
    "revision/experiments/run_abides_typed_audit_ippo_v1.py",
    "revision/experiments/setup_official_on_policy.py",
    "revision/notes/abides_typed_audit_ippo_v2_protocol.md",
    "revision/src/__init__.py",
    "revision/src/agents/__init__.py",
    "revision/src/agents/abides_mappo.py",
    "revision/src/agents/mappo.py",
    "revision/src/agents/official_on_policy_mappo.py",
    "revision/src/agents/typed_audit.py",
    "revision/src/envs/__init__.py",
    "revision/src/envs/abides_docker_env.py",
    "revision/tests/test_abides_typed_audit_ippo.py",
    "revision/tests/test_abides_typed_audit_runner.py",
    "scripts/verify_release.py",
}
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
BINARY_SUFFIXES = {".png"}


def tracked_files() -> set[str]:
    files = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == "outputs":
            continue
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise SystemExit(f"FAIL: symlink is not allowed: {relative}")
        if path.is_file() and path.name != ".DS_Store":
            files.add(relative.as_posix())
    return files


files = tracked_files()
if files != ALLOWED:
    raise SystemExit(
        "FAIL: release file set differs from the allowlist; "
        f"missing={sorted(ALLOWED - files)} extra={sorted(files - ALLOWED)}"
    )

link = re.compile(r'(?:href|src|srcset)="([^"]+)"|\]\(([^)\s]+)\)')
for readme in ("README.md", "README.zh-CN.md"):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for match in link.finditer(text):
        target = (match.group(1) or match.group(2)).split("#", 1)[0]
        if not target or re.match(r"[a-z]+:", target):
            continue
        if not (ROOT / target).exists():
            raise SystemExit(f"FAIL: broken link in {readme}: {target}")

forbidden = (
    "State" + "-Aligned",
    "Order" + "-Flow Contracts",
    "paper " + "writing",
)
absolute_markers = ("/" + "Users/", "/" + "home/", ":\\" + "Users\\")
email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
for relative in sorted(files):
    if Path(relative).suffix in BINARY_SUFFIXES:
        continue
    text = (ROOT / relative).read_text(encoding="utf-8")
    if any(marker in text for marker in absolute_markers):
        raise SystemExit(f"FAIL: absolute user path in {relative}")
    if email.search(text):
        raise SystemExit(f"FAIL: email address in {relative}")
    for token in forbidden:
        if token.casefold() in text.casefold():
            raise SystemExit(f"FAIL: cross-project token in {relative}")

result_root = (
    ROOT
    / "revision"
    / "experiments"
    / "outputs"
    / "abides_typed_audit_ippo_v2"
    / "formal"
)
summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
rows = json.loads((result_root / "rows.json").read_text(encoding="utf-8"))
state = json.loads((result_root / "run_state.json").read_text(encoding="utf-8"))
if state["status"] != "complete" or state["failed_policy_seeds"]:
    raise SystemExit("FAIL: adaptive formal run is not complete")
if state["completed_policy_seeds"] != list(range(560000, 560010)):
    raise SystemExit("FAIL: adaptive formal policy-seed ledger changed")
if len(rows) != 10:
    raise SystemExit(f"FAIL: expected 10 adaptive seed rows, got {len(rows)}")


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f"FAIL: {label} changed: {actual} != {expected}")


close(
    summary["paired_contrasts"]["full_ace_minus_vanilla_ippo"]["selectivity"]
    ["t_interval"]["mean"],
    0.4856383883208082,
    "Full minus Vanilla selectivity",
)
close(
    summary["paired_contrasts"]["full_ace_minus_always_on_actor"]["selectivity"]
    ["t_interval"]["mean"],
    0.4970416866596029,
    "Full minus always-on selectivity",
)
close(
    summary["paired_contrasts"]["full_ace_minus_vanilla_ippo"]
    ["mean_per_dealer_gross_pnl"]["t_interval"]["mean"],
    -96.7965001581704,
    "Full minus Vanilla gross PnL",
)
close(
    summary["method_summary"]["full_ace"]["mean_per_dealer_gross_pnl"]["mean"],
    1014.2575052773107,
    "Full gross PnL",
)

config = json.loads(
    (ROOT / "revision" / "configs" / "abides_typed_audit_ippo_v2.json").read_text(
        encoding="utf-8"
    )
)
protocol_hash = hashlib.sha256((ROOT / config["protocol_file"]).read_bytes()).hexdigest()
if protocol_hash != config["protocol_file_sha256"]:
    raise SystemExit(f"FAIL: frozen protocol changed: {config['protocol_file']}")

git_dir = ROOT / ".git"
if git_dir.is_dir():
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    if git("branch", "--show-current") != "main":
        raise SystemExit("FAIL: release branch must be main")
    if git("status", "--porcelain"):
        raise SystemExit("FAIL: repository has uncommitted changes")

print(f"PASS: {len(files)} allowlisted files; release checks passed")
