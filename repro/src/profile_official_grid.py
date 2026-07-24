#!/usr/bin/env python3
"""Profile the official HyperGrid worker used by the threaded full matrix."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / ".openresearch" / "artifacts" / "claim-2" / "cpu-profile"
STEPS = 100
SEED = 604


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "repro/src/run_grid_training_single_thread.py",
        "train_gfn.py",
        "--device",
        "cpu",
        "--custom_dist",
        "shubert",
        "--seed",
        str(SEED),
        "--n_iterations",
        str(STEPS),
        "--validation_interval",
        str(STEPS),
        "--save_dir",
        str(ARTIFACT_DIR),
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    (ARTIFACT_DIR / "raw_stdout.txt").write_text(proc.stdout)
    if proc.returncode:
        print(proc.stdout)
        raise SystemExit(proc.returncode)

    checkpoints = sorted(ARTIFACT_DIR.glob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one checkpoint, found {len(checkpoints)}")

    seconds_per_step = elapsed / STEPS
    # Table 1 needs eight ingredient GFNs plus four MO-GFNs and four HN-GFNs.
    projected_training_seconds = seconds_per_step * 20_000 * 16
    max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = {
        "purpose": "capacity planning only; not claim evidence",
        "worker_mode": "official entrypoint with one PyTorch CPU thread",
        "faithful_dimensions": {
            "grid": "32x32",
            "batch_size": 128,
            "hidden_dim": 64,
            "n_hidden": 2,
            "loss": "SubTB",
            "replay_buffer_size": 10_000,
        },
        "profile_deviation": {
            "training_steps": STEPS,
            "paper_code_default_steps": 20_000,
        },
        "seed": SEED,
        "command": command,
        "elapsed_seconds": elapsed,
        "seconds_per_step": seconds_per_step,
        "projected_seconds_for_16_models_at_20000_steps": projected_training_seconds,
        "projected_cpu_hours_for_16_models_at_20000_steps": (
            projected_training_seconds / 3600
        ),
        "checkpoint": {
            "name": checkpoints[0].name,
            "sha256": sha256(checkpoints[0]),
            "bytes": checkpoints[0].stat().st_size,
        },
        "platform": platform.platform(),
        "max_child_rss_platform_units": max_rss,
    }
    (ARTIFACT_DIR / "profile.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("OFFICIAL_GRID_CPU_PROFILE_JSON")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
