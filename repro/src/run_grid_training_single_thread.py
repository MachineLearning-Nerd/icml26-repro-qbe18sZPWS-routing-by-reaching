#!/usr/bin/env python3
"""Run an official grid training entrypoint with one PyTorch CPU thread."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "official" / "grid"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_grid_training_single_thread.py <entrypoint.py> [args...]"
        )
    entrypoint = GRID / sys.argv[1]
    if entrypoint.name not in {
        "train_gfn.py",
        "train_mogfn.py",
        "train_hngfn.py",
    }:
        raise SystemExit(f"unsupported training entrypoint: {entrypoint.name}")
    if not entrypoint.is_file():
        raise SystemExit(f"missing training entrypoint: {entrypoint}")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
