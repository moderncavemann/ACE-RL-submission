#!/usr/bin/env python3
"""Fail-closed structural and anonymity checks for this repository."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "ace_rl_project/__init__.py",
    "ace_rl_project/envs/__init__.py",
    "ace_rl_project/envs/lob_env.py",
    "ace_rl_project/experiments/__init__.py",
    "ace_rl_project/experiments/config.py",
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
    "scripts/verify_release.py",
}
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs"}


def tracked_files() -> set[str]:
    files = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
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

readme_headings = re.findall(
    r"^# (.+)$", (ROOT / "README.md").read_text(encoding="utf-8"), re.MULTILINE
)
if readme_headings != ["Environment", "Dependencies", "Run"]:
    raise SystemExit(f"FAIL: unexpected README sections: {readme_headings}")

forbidden = (
    "State" + "-Aligned",
    "Order" + "-Flow Contracts",
    "AB" + "IDES",
    "paper " + "writing",
    "revision" + "/",
)
absolute_markers = ("/" + "Users/", "/" + "home/", ":\\" + "Users\\")
email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
for relative in sorted(files):
    text = (ROOT / relative).read_text(encoding="utf-8")
    if any(marker in text for marker in absolute_markers):
        raise SystemExit(f"FAIL: absolute user path in {relative}")
    if email.search(text):
        raise SystemExit(f"FAIL: email address in {relative}")
    for token in forbidden:
        if token.casefold() in text.casefold():
            raise SystemExit(f"FAIL: cross-project token in {relative}")

git_dir = ROOT / ".git"
if git_dir.is_dir():
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    if git("rev-list", "--count", "HEAD") != "1":
        raise SystemExit("FAIL: anonymous repository must have exactly one commit")
    if git("branch", "--show-current") != "main":
        raise SystemExit("FAIL: anonymous repository branch must be main")
    if git("tag", "--list"):
        raise SystemExit("FAIL: anonymous repository must not contain tags")
    if git("show", "-s", "--format=%an <%ae>", "HEAD") != (
        "Anonymous Authors <anonymous@invalid>"
    ):
        raise SystemExit("FAIL: commit identity is not anonymous")
    if git("status", "--porcelain"):
        raise SystemExit("FAIL: anonymous repository has uncommitted changes")

print(f"PASS: {len(files)} allowlisted files; anonymity checks passed")
