#!/usr/bin/env python3
"""Train and evaluate the full seeded HyperGrid claim matrix."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "official" / "grid"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
OUTPUT = ARTIFACTS / "claim-2" / "full_grid_raw.json"
TRAIN_ROOT = ARTIFACTS / "claim-2" / "trained"
SEEDS = [1337]
N_ITERATIONS = 20_000
N_PREFERENCES = 128
TRAIN_WORKERS = 8
BASE_REWARDS = [
    "shubert",
    "diagonal",
    "currin",
    "sphere",
    "branin",
    "circle1",
    "circle2",
    "circle3",
]
MIX_REWARDS = {k: f"mix{k}" for k in range(2, 6)}
PAPER_TABLE = {
    "ours": {2: 0.003, 3: 0.003, 4: 0.003, 5: 0.003},
    "mogfn": {2: 0.021, 3: 0.027, 4: 0.042, 5: 0.048},
    "hngfn": {2: 0.017, 3: 0.021, 4: 0.032, 5: 0.035},
    "ensemble": {2: 0.117, 3: 0.098, 4: 0.113, 5: 0.111},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_specs() -> list[dict]:
    specs = []
    for seed in SEEDS:
        for reward in BASE_REWARDS:
            specs.append(
                {
                    "seed": seed,
                    "kind": "ingredient",
                    "name": reward,
                    "script": "train_gfn.py",
                    "extra": ["--custom_dist", reward],
                }
            )
        for k, reward in MIX_REWARDS.items():
            common = [
                "--custom_dist",
                reward,
                "--n_reward_fns",
                str(k),
                "--n_simplex_for_val",
                str(N_PREFERENCES),
            ]
            specs.append(
                {
                    "seed": seed,
                    "kind": "mogfn",
                    "name": f"mix{k}",
                    "script": "train_mogfn.py",
                    "extra": common,
                }
            )
            specs.append(
                {
                    "seed": seed,
                    "kind": "hngfn",
                    "name": f"mix{k}",
                    "script": "train_hngfn.py",
                    "extra": common,
                }
            )
    return specs


def train_one(spec: dict) -> dict:
    output_dir = TRAIN_ROOT / f"seed-{spec['seed']}" / spec["kind"] / spec["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "repro" / "src" / "run_grid_training_single_thread.py"),
        spec["script"],
        "--device",
        "cpu",
        "--seed",
        str(spec["seed"]),
        "--n_iterations",
        str(N_ITERATIONS),
        "--validation_interval",
        str(N_ITERATIONS),
        "--save_dir",
        str(output_dir),
        *spec["extra"],
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
    (output_dir / "training_stdout.txt").write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(
            f"{spec['kind']} {spec['name']} seed {spec['seed']} failed "
            f"with exit {proc.returncode}; tail:\n{proc.stdout[-4000:]}"
        )
    checkpoints = sorted(output_dir.glob("*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"{spec['kind']} {spec['name']} seed {spec['seed']}: "
            f"expected one checkpoint, found {len(checkpoints)}"
        )
    return {
        **spec,
        "command": command,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(checkpoints[0].relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoints[0]),
        "checkpoint_bytes": checkpoints[0].stat().st_size,
    }


def aggregate(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "n_seeds": 1,
            "mean": mean,
            "sample_std": None,
            "ci95_t": None,
            "per_seed": values,
            "partial_seed_shard": True,
        }
    std = statistics.stdev(values)
    half_width = 4.302652729696142 * std / math.sqrt(len(values))
    return {
        "n_seeds": len(values),
        "mean": mean,
        "sample_std": std,
        "ci95_t": [mean - half_width, mean + half_width],
        "per_seed": values,
    }


def checkpoint_path(seed: int, kind: str, name: str) -> Path:
    paths = sorted((TRAIN_ROOT / f"seed-{seed}" / kind / name).glob("*.pt"))
    if len(paths) != 1:
        raise RuntimeError(f"checkpoint lookup failed for {seed}/{kind}/{name}")
    return paths[0]


def load_grid_modules() -> dict:
    sys.path.insert(0, str(GRID))
    from gfn.preprocessors import KHotPreprocessor
    from src.advanced_hypergrid import (
        AdvancedConditionalHyperGrid,
        AdvancedHyperGrid,
    )
    from src.mixture_estimator import MixtureDiscretePolicyEstimator
    from src.training import evaluate_conditional
    from src.utils import (
        get_exact_u_and_P_T,
        set_up_conditional_gflownet,
        set_up_gflownet,
        set_up_hn_gflownet,
    )

    return locals()


def load_base_models(modules: dict, env, seed: int, rewards: list[str]):
    preprocessor = modules["KHotPreprocessor"](height=env.height, ndim=env.ndim)
    models = []
    for reward in rewards:
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
                checkpoint_path(seed, "ingredient", reward),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        models.append(model)
    return preprocessor, models


def evaluate_mixing(modules: dict, seed: int, k: int, mixing_type: str) -> list[float]:
    env = modules["AdvancedHyperGrid"](
        ndim=2,
        height=32,
        n_reward_fns=k,
        device="cpu",
        custom_dist=f"mix{k}",
        beta=1.0,
    )
    rewards = BASE_REWARDS[:k]
    preprocessor, models = load_base_models(modules, env, seed, rewards)
    from src.simplex import simplex_list

    errors = []
    for condition in simplex_list(k, N_PREFERENCES, "cpu", seed=0):
        env.set_conditioning(condition)
        estimator = modules["MixtureDiscretePolicyEstimator"](
            gfn_list=models,
            n_actions=env.n_actions,
            preprocessor=preprocessor,
            loss_type="SubTB",
            is_backward=False,
            mixing_type=mixing_type,
            env=env,
            weight_list=condition.squeeze(0).tolist(),
            beta=1.0,
            flow_estimation="logF",
        )
        _, learned = modules["get_exact_u_and_P_T"](env, estimator)
        errors.append(float((learned - env.true_dist.cpu()).abs().sum().item()))
    return errors


def evaluate_conditional_model(
    modules: dict, seed: int, k: int, kind: str
) -> list[float]:
    env = modules["AdvancedConditionalHyperGrid"](
        ndim=2,
        height=32,
        n_reward_fns=k,
        device="cpu",
        custom_dist=f"mix{k}",
    )
    preprocessor = modules["KHotPreprocessor"](height=env.height, ndim=env.ndim)
    if kind == "mogfn":
        model = modules["set_up_conditional_gflownet"](
            env,
            64,
            2,
            preprocessor,
            "SubTB",
            k,
            16,
            "geometric_within",
            2.0,
        )
    elif kind == "hngfn":
        model = modules["set_up_hn_gflownet"](
            env=env,
            preprocessor=preprocessor,
            n_reward_fns=k,
            ray_hidden_dim=32,
            hidden_dim=64,
            n_hidden=2,
            subTB_weighting="geometric_within",
            subTB_lambda=2.0,
            predictor_layers=[],
            logit_clipping=0.0,
        )
    else:
        raise ValueError(kind)
    model.load_state_dict(
        torch.load(
            checkpoint_path(seed, kind, f"mix{k}"),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    result = modules["evaluate_conditional"](
        env,
        model,
        k,
        N_PREFERENCES,
        result_saved_path=None,
    )
    return [float(value) for value in result["l1_errors"]]


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def distortion_setting(
    modules: dict,
    seed: int,
    rewards: list[str],
    custom_dist: str,
    mixing_type: str,
) -> dict:
    env = modules["AdvancedHyperGrid"](
        ndim=2,
        height=32,
        device="cpu",
        custom_dist=custom_dist,
    )
    preprocessor, models = load_base_models(modules, env, seed, rewards)
    estimator = modules["MixtureDiscretePolicyEstimator"](
        gfn_list=models,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
        loss_type="SubTB",
        is_backward=False,
        mixing_type=mixing_type,
        env=env,
        weight_list=[1 / len(models)] * len(models),
        beta=1.0,
        flow_estimation="logF",
    )
    mixed_reaching_t, induced_t = modules["get_exact_u_and_P_T"](
        env, estimator
    )
    ingredient_t = [
        modules["get_exact_u_and_P_T"](env, model.pf)[1] for model in models
    ]
    states = env.all_states
    model_action_probabilities = []
    with torch.no_grad():
        for model in models:
            output = model.pf(states)
            distribution = model.pf.to_probability_distribution(
                states=states,
                module_output=output,
                epsilon=0.0,
            )
            model_action_probabilities.append(distribution.probs.detach().cpu())
    action_probabilities = torch.stack(model_action_probabilities)
    model_reaching = estimator.us.reshape(len(models), -1).detach().cpu()
    weighted_actions = model_reaching.unsqueeze(-1) * action_probabilities
    if mixing_type == "harmonic_mean":
        numerator = torch.stack(ingredient_t).prod(dim=0)
        denominator = sum(
            numerator / (component + 1e-30) for component in ingredient_t
        )
        induced_component_composition = numerator / (denominator + 1e-30)
        action_numerator = weighted_actions.prod(dim=0)
        action_denominator = sum(
            action_numerator / (weighted_actions[index] + 1e-10)
            for index in range(len(models))
        )
        unnormalized_action_scores = action_numerator / (
            action_denominator + 1e-10
        )
    elif mixing_type == "contrast":
        if len(ingredient_t) != 2:
            raise ValueError("registered distortion contrasts are two-way")
        induced_component_composition = ingredient_t[0] ** 2 / (
            ingredient_t[0] + ingredient_t[1] + 1e-30
        )
        unnormalized_action_scores = weighted_actions[0] ** 2 / (
            weighted_actions.sum(dim=0) + 1e-10
        )
    else:
        raise ValueError(mixing_type)

    # Eq. (5)'s terminal-action numerator is exactly G(p_1(x),...,p_k(x))
    # for the Model-F reaching estimates used by the mixing policy.
    composition = unnormalized_action_scores[:, -1]
    g = composition.detach().cpu().numpy().astype(np.float64)
    induced_component_g = (
        induced_component_composition.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    induced = induced_t.detach().cpu().numpy().astype(np.float64)
    local_normalizer = (
        unnormalized_action_scores.sum(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    mixed_reaching = (
        mixed_reaching_t.detach().cpu().numpy().astype(np.float64)
    )
    z = float(g.sum())
    reference = 1.0 / z
    delta = mixed_reaching / np.maximum(local_normalizer, 1e-300)
    identity_delta = induced / np.maximum(g, 1e-300)
    relative_deviation = np.abs(delta - reference) / reference
    order = np.argsort(g)
    low = order[: len(order) // 2]
    high = order[-max(1, len(order) // 10) :]
    q1, q3 = np.quantile(delta, [0.25, 0.75])
    iqr = q3 - q1
    outliers = (delta < q1 - 1.5 * iqr) | (delta > q3 + 1.5 * iqr)
    bins = []
    for index, indices in enumerate(np.array_split(order, 10)):
        bins.append(
            {
                "g_decile": index + 1,
                "g_min": float(g[indices].min()),
                "g_max": float(g[indices].max()),
                "median_relative_deviation": float(
                    np.median(relative_deviation[indices])
                ),
                "target_mass": float(g[indices].sum() / z),
            }
        )
    target = g / z
    l1_contribution = np.abs(induced - target)
    return {
        "seed": seed,
        "rewards": rewards,
        "custom_dist": custom_dist,
        "mixing_type": mixing_type,
        "states": len(g),
        "z_m": z,
        "one_over_z_m": reference,
        "induced_mass": float(induced.sum()),
        "target_mass": float(target.sum()),
        "l1": float(l1_contribution.sum()),
        "model_f_vs_rollout_component_target_l1": float(
            np.abs(
                g / g.sum()
                - induced_component_g / induced_component_g.sum()
            ).sum()
        ),
        "delta_q1": float(q1),
        "delta_q3": float(q3),
        "delta_iqr": float(iqr),
        "delta_identity_max_abs_residual": float(
            np.max(np.abs(delta - identity_delta))
        ),
        "high_g_median_relative_deviation": float(
            np.median(relative_deviation[high])
        ),
        "low_g_median_relative_deviation": float(
            np.median(relative_deviation[low])
        ),
        "high_g_target_mass": float(g[high].sum() / z),
        "high_g_l1_share": float(
            l1_contribution[high].sum() / max(l1_contribution.sum(), 1e-300)
        ),
        "outlier_count": int(outliers.sum()),
        "outlier_target_mass": float(g[outliers].sum() / z),
        "outliers_in_low_half_fraction": float(
            np.logical_and(outliers, np.isin(np.arange(len(g)), low)).sum()
            / max(outliers.sum(), 1)
        ),
        "spearman_g_vs_relative_deviation": rank_correlation(
            g, relative_deviation
        ),
        "deciles": bins,
    }


def distortion_specs() -> list[tuple[list[str], str, str]]:
    return [
        (["shubert", "diagonal"], "harmonic_mean_shu_diag", "harmonic_mean"),
        (["shubert", "diagonal"], "contrast_shu_diag", "contrast"),
        (["diagonal", "shubert"], "contrast_diag_shu", "contrast"),
        (["shubert", "sphere"], "harmonic_mean_shu_sph", "harmonic_mean"),
        (["shubert", "sphere"], "contrast_shu_sph", "contrast"),
        (["sphere", "shubert"], "contrast_sph_shu", "contrast"),
        (["circle1", "circle2"], "harmonic_mean_circle12", "harmonic_mean"),
        (["circle1", "circle2"], "contrast_circle12", "contrast"),
        (["circle2", "circle1"], "contrast_circle21", "contrast"),
        (["circle2", "circle3"], "harmonic_mean_circle23", "harmonic_mean"),
        (["circle2", "circle3"], "contrast_circle23", "contrast"),
        (["circle3", "circle2"], "contrast_circle32", "contrast"),
    ]


def summarize(raw: dict) -> dict:
    aggregates = {}
    for method in ("ours", "ensemble", "mogfn", "hngfn"):
        aggregates[method] = {}
        for k in range(2, 6):
            seed_means = [
                statistics.fmean(raw["table1"][str(seed)][method][str(k)])
                for seed in SEEDS
            ]
            aggregates[method][str(k)] = aggregate(seed_means)

    claim_2_checks = {
        "three_deterministic_seeds": len(raw["table1"]) == 3,
        "exactly_128_preferences_per_cell": all(
            len(raw["table1"][str(seed)][method][str(k)]) == N_PREFERENCES
            for seed in SEEDS
            for method in ("ours", "ensemble", "mogfn", "hngfn")
            for k in range(2, 6)
        ),
        "ours_below_0_01_each_k": all(
            aggregates["ours"][str(k)]["mean"] <= 0.01 for k in range(2, 6)
        ),
        "ours_beats_both_trained_baselines_every_seed_and_k": all(
            statistics.fmean(raw["table1"][str(seed)]["ours"][str(k)])
            < min(
                statistics.fmean(raw["table1"][str(seed)]["mogfn"][str(k)]),
                statistics.fmean(raw["table1"][str(seed)]["hngfn"][str(k)]),
            )
            for seed in SEEDS
            for k in range(2, 6)
        ),
        "trained_baselines_within_0_02_of_paper": all(
            abs(
                aggregates[method][str(k)]["mean"]
                - PAPER_TABLE[method][k]
            )
            <= 0.02
            for method in ("mogfn", "hngfn")
            for k in range(2, 6)
        ),
    }
    claim_5_checks = {
        "paired_improvement_at_least_0_05_every_seed_and_k": all(
            statistics.fmean(raw["table1"][str(seed)]["ensemble"][str(k)])
            - statistics.fmean(raw["table1"][str(seed)]["ours"][str(k)])
            >= 0.05
            for seed in SEEDS
            for k in range(2, 6)
        ),
        "ensemble_within_0_03_of_paper": all(
            abs(
                aggregates["ensemble"][str(k)]["mean"]
                - PAPER_TABLE["ensemble"][k]
            )
            <= 0.03
            for k in range(2, 6)
        ),
    }
    distortions = raw["distortion"]
    claim_6_checks = {
        "all_12_figure_a6_settings_each_seed": len(distortions) == 36,
        "mass_and_state_sanity": all(
            row["states"] == 1024
            and abs(row["induced_mass"] - 1.0) < 1e-5
            and abs(row["target_mass"] - 1.0) < 1e-12
            and row["delta_identity_max_abs_residual"] < 1e-4
            for row in distortions
        ),
        "high_g_closer_than_low_g_every_setting": all(
            row["high_g_median_relative_deviation"]
            < row["low_g_median_relative_deviation"]
            for row in distortions
        ),
        "high_g_median_relative_deviation_at_most_0_25": all(
            row["high_g_median_relative_deviation"] <= 0.25
            for row in distortions
        ),
    }
    return {
        "paper_table": {
            method: {str(k): value for k, value in values.items()}
            for method, values in PAPER_TABLE.items()
        },
        "aggregates": aggregates,
        "claim_2": {
            "verdict": (
                "VERIFIED" if all(claim_2_checks.values()) else "BLOCKED"
            ),
            "checks": claim_2_checks,
        },
        "claim_5": {
            "verdict": (
                "VERIFIED" if all(claim_5_checks.values()) else "BLOCKED"
            ),
            "checks": claim_5_checks,
        },
        "claim_6": {
            "verdict": (
                "VERIFIED" if all(claim_6_checks.values()) else "BLOCKED"
            ),
            "checks": claim_6_checks,
            "high_g_median_relative_deviation_range": [
                min(
                    row["high_g_median_relative_deviation"]
                    for row in distortions
                ),
                max(
                    row["high_g_median_relative_deviation"]
                    for row in distortions
                ),
            ],
            "low_g_median_relative_deviation_range": [
                min(
                    row["low_g_median_relative_deviation"]
                    for row in distortions
                ),
                max(
                    row["low_g_median_relative_deviation"]
                    for row in distortions
                ),
            ],
        },
    }


def main() -> None:
    campaign_start = time.perf_counter()
    specs = training_specs()
    max_workers = min(TRAIN_WORKERS, max(1, os.cpu_count() or 1))
    print(
        f"Training {len(specs)} faithful models with {max_workers} "
        "concurrent single-threaded CPU workers",
        flush=True,
    )
    training = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(train_one, spec): spec for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            training.append(result)
            print(
                "TRAINED "
                f"seed={result['seed']} kind={result['kind']} "
                f"name={result['name']} seconds={result['elapsed_seconds']:.2f} "
                f"sha256={result['checkpoint_sha256']}",
                flush=True,
            )

    modules = load_grid_modules()
    raw = {
        "contract": {
            "paper": "arXiv:2602.21565v1",
            "grid": "32x32",
            "training_iterations": N_ITERATIONS,
            "preferences_per_k": N_PREFERENCES,
            "seeds": SEEDS,
            "training_models": len(specs),
            "evaluation": "exact dynamic-programming terminal distributions",
        },
        "training": sorted(
            training, key=lambda row: (row["seed"], row["kind"], row["name"])
        ),
        "table1": {},
        "distortion": [],
    }
    for seed in SEEDS:
        print(f"EVALUATING_TABLE1 seed={seed}", flush=True)
        raw["table1"][str(seed)] = {
            method: {} for method in ("ours", "ensemble", "mogfn", "hngfn")
        }
        for k in range(2, 6):
            raw["table1"][str(seed)]["ours"][str(k)] = evaluate_mixing(
                modules, seed, k, "scalarization"
            )
            raw["table1"][str(seed)]["ensemble"][str(k)] = evaluate_mixing(
                modules, seed, k, "scalarization_without_u"
            )
            raw["table1"][str(seed)]["mogfn"][str(k)] = (
                evaluate_conditional_model(modules, seed, k, "mogfn")
            )
            raw["table1"][str(seed)]["hngfn"][str(k)] = (
                evaluate_conditional_model(modules, seed, k, "hngfn")
            )
            print(
                "TABLE1_CELL "
                f"seed={seed} k={k} "
                + " ".join(
                    f"{method}="
                    f"{statistics.fmean(raw['table1'][str(seed)][method][str(k)]):.6f}"
                    for method in ("ours", "ensemble", "mogfn", "hngfn")
                ),
                flush=True,
            )
        for rewards, custom_dist, mixing_type in distortion_specs():
            row = distortion_setting(
                modules, seed, rewards, custom_dist, mixing_type
            )
            raw["distortion"].append(row)
            print(
                "DISTORTION "
                f"seed={seed} setting={custom_dist} "
                f"high={row['high_g_median_relative_deviation']:.6f} "
                f"low={row['low_g_median_relative_deviation']:.6f}",
                flush=True,
            )

    raw["elapsed_seconds"] = time.perf_counter() - campaign_start
    raw["summary"] = summarize(raw)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")

    checker = subprocess.run(
        [
            sys.executable,
            "repro/src/verify_full_grid_claims.py",
            "--input",
            str(OUTPUT),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (ARTIFACTS / "claim-2" / "full_grid_checker.txt").write_text(
        checker.stdout
    )
    if checker.returncode:
        print(checker.stdout)
        raise RuntimeError("independent full-grid checker failed")

    negative = subprocess.run(
        [
            sys.executable,
            "repro/src/verify_full_grid_claims.py",
            "--input",
            str(OUTPUT),
            "--negative-control",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (ARTIFACTS / "claim-2" / "full_grid_negative_control.txt").write_text(
        negative.stdout
    )
    if negative.returncode == 0:
        raise RuntimeError("negative control unexpectedly passed")

    print("FULL_GRID_CHECKER")
    print(checker.stdout)
    print("FULL_GRID_NEGATIVE_CONTROL_EXPECTED_FAILURE")
    print(negative.stdout)
    print("FULL_GRID_RAW_JSON")
    print(json.dumps(raw, sort_keys=True))


if __name__ == "__main__":
    main()
