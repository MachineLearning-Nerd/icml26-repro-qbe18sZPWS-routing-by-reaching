#!/usr/bin/env python3
"""Regenerate direct state-level evidence for the nonlinear distortion claim."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "official" / "grid"
CHECKPOINT_ROOT = ROOT / "repro" / "evidence" / "claim-6" / "checkpoints"
OUTPUT = ROOT / "evidence" / "claim-6" / "direct_state_audit.json"
PARENT_RAW = ROOT / "repro" / "evidence" / "full_grid_raw.json"
SEEDS = [604, 1337, 20260719]
REWARDS = ["circle1", "circle2"]
N_ITERATIONS = 20_000
TRAIN_WORKERS = 4
OPERATORS = [
    ("harmonic_mean_circle12", "harmonic_mean"),
    ("contrast_circle12", "contrast"),
]
OFFICIAL_SOURCE_COMMIT = "b82493c8cd9b46a0933ab8f19440aebbf3e14b28"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(seed: int, reward: str) -> Path:
    directory = CHECKPOINT_ROOT / f"seed-{seed}" / reward
    checkpoints = sorted(directory.glob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected one checkpoint for seed={seed} reward={reward}, "
            f"found {len(checkpoints)}"
        )
    return checkpoints[0]


def train_one(seed: int, reward: str, parent: dict) -> dict:
    output_dir = CHECKPOINT_ROOT / f"seed-{seed}" / reward
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.pt"))
    elapsed = 0.0
    reused = len(existing) == 1
    command = [
        sys.executable,
        "repro/src/run_grid_training_single_thread.py",
        "train_gfn.py",
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--n_iterations",
        str(N_ITERATIONS),
        "--validation_interval",
        str(N_ITERATIONS),
        "--save_dir",
        str(output_dir),
        "--custom_dist",
        reward,
    ]
    if not existing:
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
        (output_dir / "training_stdout.txt").write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(
                f"training failed for seed={seed} reward={reward}:\n"
                f"{proc.stdout[-4000:]}"
            )
    path = checkpoint_path(seed, reward)
    expected = next(
        row
        for row in parent["training"]
        if row["seed"] == seed
        and row["kind"] == "ingredient"
        and row["name"] == reward
    )
    return {
        "seed": seed,
        "reward": reward,
        "steps": N_ITERATIONS,
        "reused": reused,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(path.relative_to(ROOT)),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256(path),
        "parent_checkpoint_sha256": expected["checkpoint_sha256"],
        "byte_identical_to_parent_checkpoint": (
            sha256(path) == expected["checkpoint_sha256"]
        ),
        "command": [
            "python",
            "repro/src/run_grid_training_single_thread.py",
            "train_gfn.py",
            "--device",
            "cpu",
            "--seed",
            str(seed),
            "--n_iterations",
            str(N_ITERATIONS),
            "--validation_interval",
            str(N_ITERATIONS),
            "--save_dir",
            str(output_dir.relative_to(ROOT)),
            "--custom_dist",
            reward,
        ],
    }


def load_grid_modules() -> dict:
    sys.path.insert(0, str(GRID))
    from gfn.preprocessors import KHotPreprocessor
    from src.advanced_hypergrid import AdvancedHyperGrid
    from src.mixture_estimator import MixtureDiscretePolicyEstimator
    from src.utils import get_exact_u_and_P_T, set_up_gflownet

    return locals()


def load_models(modules: dict, env, seed: int):
    preprocessor = modules["KHotPreprocessor"](height=env.height, ndim=env.ndim)
    models = []
    for reward in REWARDS:
        model = modules["set_up_gflownet"](
            env,
            64,
            2,
            preprocessor,
            "SubTB",
            "geometric_within",
            2.0,
            logF_n_hidden=1,
        ).to("cpu")
        model.load_state_dict(
            torch.load(
                checkpoint_path(seed, reward),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        models.append(model)
    return preprocessor, models


def region_stats(
    indices: np.ndarray,
    g: np.ndarray,
    induced: np.ndarray,
    delta: np.ndarray,
    reference: float,
) -> dict:
    normalized = delta[indices] / reference
    target = g / g.sum()
    l1 = np.abs(induced - target)
    mean = float(normalized.mean())
    variance = float(normalized.var())
    std = math.sqrt(variance)
    return {
        "count": int(len(indices)),
        "g_min": float(g[indices].min()),
        "g_max": float(g[indices].max()),
        "normalized_delta_mean": mean,
        "normalized_delta_variance": variance,
        "normalized_delta_std": std,
        "normalized_delta_cv": std / max(abs(mean), 1e-300),
        "median_abs_relative_deviation": float(
            np.median(np.abs(normalized - 1.0))
        ),
        "rmse_relative_deviation": float(
            np.sqrt(np.mean(np.square(normalized - 1.0)))
        ),
        "target_mass": float(target[indices].sum()),
        "l1_error_share": float(
            l1[indices].sum() / max(float(l1.sum()), 1e-300)
        ),
    }


def audit_operator(
    modules: dict,
    seed: int,
    custom_dist: str,
    mixing_type: str,
) -> dict:
    env = modules["AdvancedHyperGrid"](
        ndim=2,
        height=32,
        device="cpu",
        custom_dist=custom_dist,
    )
    preprocessor, models = load_models(modules, env, seed)
    estimator = modules["MixtureDiscretePolicyEstimator"](
        gfn_list=models,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
        loss_type="SubTB",
        is_backward=False,
        mixing_type=mixing_type,
        env=env,
        weight_list=[0.5, 0.5],
        beta=1.0,
        flow_estimation="logF",
    )
    u_m_t, induced_t = modules["get_exact_u_and_P_T"](env, estimator)
    states = env.all_states
    action_probabilities = []
    with torch.no_grad():
        for model in models:
            output = model.pf(states)
            distribution = model.pf.to_probability_distribution(
                states=states,
                module_output=output,
                epsilon=0.0,
            )
            action_probabilities.append(distribution.probs.detach().cpu())
    actions = torch.stack(action_probabilities)
    reaching = estimator.us.reshape(len(models), -1).detach().cpu()
    weighted_actions = reaching.unsqueeze(-1) * actions
    if mixing_type == "harmonic_mean":
        numerator = weighted_actions.prod(dim=0)
        denominator = sum(
            numerator / (weighted_actions[index] + 1e-10)
            for index in range(len(models))
        )
        scores = numerator / (denominator + 1e-10)
    elif mixing_type == "contrast":
        scores = weighted_actions[0] ** 2 / (
            weighted_actions.sum(dim=0) + 1e-10
        )
    else:
        raise ValueError(mixing_type)

    g = scores[:, -1].detach().cpu().numpy().astype(np.float64)
    n_m = scores.sum(dim=-1).detach().cpu().numpy().astype(np.float64)
    u_m = u_m_t.detach().cpu().numpy().astype(np.float64)
    induced = induced_t.detach().cpu().numpy().astype(np.float64)
    z_m = float(g.sum())
    reference = 1.0 / z_m
    delta = u_m / np.maximum(n_m, 1e-300)
    target = g / z_m
    order = np.argsort(g, kind="stable")
    low = order[: len(order) // 2]
    high = order[-max(1, len(order) // 10) :]
    all_states = np.arange(len(g))
    equation_9 = np.abs(delta - reference) * g
    direct_l1 = np.abs(induced - target)
    return {
        "seed": seed,
        "operator": custom_dist,
        "mixing_type": mixing_type,
        "states": len(g),
        "z_m": z_m,
        "one_over_z_m": reference,
        "induced_mass": float(induced.sum()),
        "target_mass": float(target.sum()),
        "l1": float(direct_l1.sum()),
        "delta_identity_max_abs_residual": float(
            np.max(np.abs(delta - induced / np.maximum(g, 1e-300)))
        ),
        "equation_9_max_abs_residual": float(
            np.max(np.abs(equation_9 - direct_l1))
        ),
        "regions": {
            "all_states": region_stats(
                all_states, g, induced, delta, reference
            ),
            "bottom_half_g": region_stats(low, g, induced, delta, reference),
            "high_g_decile": region_stats(high, g, induced, delta, reference),
        },
        "state_values": {
            "state_index": all_states.tolist(),
            "g": g.tolist(),
            "u_m": u_m.tolist(),
            "n_m": n_m.tolist(),
            "delta": delta.tolist(),
            "induced_probability": induced.tolist(),
        },
    }


def parent_comparison(parent: dict, row: dict) -> dict:
    old = next(
        candidate
        for candidate in parent["distortion"]
        if candidate["seed"] == row["seed"]
        and candidate["custom_dist"] == row["operator"]
    )
    high = row["regions"]["high_g_decile"]
    low = row["regions"]["bottom_half_g"]
    current = {
        "high_g_median_relative_deviation": high[
            "median_abs_relative_deviation"
        ],
        "low_g_median_relative_deviation": low[
            "median_abs_relative_deviation"
        ],
        "high_g_target_mass": high["target_mass"],
        "high_g_l1_share": high["l1_error_share"],
    }
    return {
        key: {
            "parent": old[key],
            "rerun": value,
            "absolute_difference": abs(old[key] - value),
        }
        for key, value in current.items()
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    parent = json.loads(PARENT_RAW.read_text())
    training_start = time.perf_counter()
    training = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=TRAIN_WORKERS
    ) as pool:
        futures = {
            pool.submit(train_one, seed, reward, parent): (seed, reward)
            for seed in SEEDS
            for reward in REWARDS
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            training.append(row)
            print(
                "CHECKPOINT "
                f"seed={row['seed']} reward={row['reward']} "
                f"seconds={row['elapsed_seconds']:.2f} "
                f"sha256={row['checkpoint_sha256']}",
                flush=True,
            )
    modules = load_grid_modules()
    rows = []
    for seed in SEEDS:
        for custom_dist, mixing_type in OPERATORS:
            row = audit_operator(modules, seed, custom_dist, mixing_type)
            row["parent_summary_comparison"] = parent_comparison(parent, row)
            rows.append(row)
            high = row["regions"]["high_g_decile"]
            low = row["regions"]["bottom_half_g"]
            print(
                "DIRECT_DELTA "
                f"seed={seed} operator={custom_dist} "
                f"high_variance={high['normalized_delta_variance']:.6g} "
                f"low_variance={low['normalized_delta_variance']:.6g} "
                f"high_rmse={high['rmse_relative_deviation']:.6g} "
                f"low_rmse={low['rmse_relative_deviation']:.6g}",
                flush=True,
            )
    result = {
        "schema_version": 1,
        "paper": "arXiv:2602.21565v1",
        "openreview_id": "qbe18sZPWS",
        "claim": 6,
        "source_anchor": "Section 5.3, Figure 5, Equations 6 and 9",
        "contract": (
            "For each of the two primary Figure 5 operators and three fixed "
            "seeds, enumerate all 1,024 terminal states and directly record "
            "G(x), u_M(x), N_M(x), and delta(x)=u_M(x)/N_M(x). The high-G "
            "decile must have lower normalized-delta variance, median absolute "
            "deviation, and RMSE than the bottom half, and its L1-error share "
            "must be below its target-mass share."
        ),
        "high_g_definition": (
            "The 102 states with largest G among all 1,024 terminal states; "
            "ties are resolved by stable state-index order."
        ),
        "low_g_comparator": "The 512 states with smallest G.",
        "grid": "32x32",
        "training_iterations": N_ITERATIONS,
        "seeds": SEEDS,
        "operators": [name for name, _ in OPERATORS],
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "parent_raw_sha256": sha256(PARENT_RAW),
        "generator_sha256": sha256(Path(__file__)),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "training_elapsed_seconds": time.perf_counter() - training_start,
        "training": sorted(training, key=lambda row: (row["seed"], row["reward"])),
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote={OUTPUT.relative_to(ROOT)} sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
