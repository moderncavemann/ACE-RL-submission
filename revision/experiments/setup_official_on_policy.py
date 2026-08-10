#!/usr/bin/env python3
"""Fetch and verify the pinned official marlbenchmark/on-policy source."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/marlbenchmark/on-policy.git"
COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
DEFAULT_DESTINATION = Path.home() / ".cache" / "ace-rl" / "on-policy" / COMMIT


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def setup(destination: Path) -> dict[str, str]:
    destination = destination.expanduser().resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", REPOSITORY, str(destination)],
            check=True,
        )
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"official source destination is not a git clone: {destination}")
    git("fetch", "--quiet", "origin", COMMIT, cwd=destination)
    git("checkout", "--quiet", "--detach", COMMIT, cwd=destination)
    resolved = git("rev-parse", "HEAD", cwd=destination)
    if resolved != COMMIT:
        raise RuntimeError(f"official source commit mismatch: {resolved} != {COMMIT}")
    if git("status", "--porcelain", cwd=destination):
        raise RuntimeError("official source checkout is dirty")
    return {
        "repository": REPOSITORY,
        "commit": resolved,
        "path": str(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(setup(parse_args().destination), sort_keys=True))


if __name__ == "__main__":
    main()
