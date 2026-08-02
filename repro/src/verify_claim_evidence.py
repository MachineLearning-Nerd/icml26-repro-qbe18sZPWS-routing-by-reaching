#!/usr/bin/env python3
"""Independent, fail-closed adjudicator for Claims 2, 5, and 6.

The input is the immutable output of the completed full neural HyperGrid run.
Each claim has a separate contract and a claim-specific negative mutation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


SEEDS = [604, 1337, 20260719]
METHODS = ("ours", "ensemble", "mogfn", "hngfn")
KS = range(2, 6)
EXPECTED_SHA256 = (
    "d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec"
)
PAPER_OURS = 0.003
T_CRITICAL_DF2_95 = 4.302652729911275
PRIMARY_FIGURE_5 = {"harmonic_mean_circle12", "contrast_circle12"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_means(raw: dict, method: str, k: int) -> list[float]:
    return [
        statistics.fmean(raw["table1"][str(seed)][method][str(k)])
        for seed in SEEDS
    ]


def aggregate(raw: dict, method: str, k: int) -> dict:
    values = seed_means(raw, method, k)
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    half_width = T_CRITICAL_DF2_95 * sample_std / math.sqrt(len(values))
    return {
        "mean": mean,
        "sample_std": sample_std,
        "ci95_t": [mean - half_width, mean + half_width],
        "per_seed": values,
    }


def common_checks(raw: dict, input_sha256: str) -> tuple[dict, dict]:
    recomputed = {
        method: {str(k): aggregate(raw, method, k) for k in KS}
        for method in METHODS
    }
    recorded = raw["summary"]["aggregates"]
    checks = {
        "immutable_parent_raw_sha256": input_sha256 == EXPECTED_SHA256,
        "paper_v1_contract": raw["contract"]["paper"] == "arXiv:2602.21565v1",
        "paper_scale_32x32": raw["contract"]["grid"] == "32x32",
        "paper_training_20000_steps": raw["contract"]["training_iterations"] == 20000,
        "all_48_models_recorded": (
            raw["contract"]["training_models"] == 48
            and len(raw["training"]) == 48
        ),
        "three_exact_seeds": (
            raw["contract"]["seeds"] == SEEDS
            and sorted(map(int, raw["table1"])) == SEEDS
        ),
        "128_preferences_every_cell": all(
            len(raw["table1"][str(seed)][method][str(k)]) == 128
            for seed in SEEDS
            for method in METHODS
            for k in KS
        ),
        "all_l1_in_valid_range": all(
            0.0 <= value <= 2.0
            for seed in SEEDS
            for method in METHODS
            for k in KS
            for value in raw["table1"][str(seed)][method][str(k)]
        ),
        "recorded_aggregates_recompute": all(
            abs(recomputed[method][str(k)]["mean"] - recorded[method][str(k)]["mean"])
            < 1e-12
            and abs(
                recomputed[method][str(k)]["sample_std"]
                - recorded[method][str(k)]["sample_std"]
            )
            < 1e-12
            and all(
                abs(a - b) < 1e-10
                for a, b in zip(
                    recomputed[method][str(k)]["ci95_t"],
                    recorded[method][str(k)]["ci95_t"],
                    strict=True,
                )
            )
            for method in METHODS
            for k in KS
        ),
    }
    return checks, recomputed


def claim_2_checks(raw: dict, recomputed: dict) -> tuple[dict, dict]:
    paper = raw["summary"]["paper_table"]
    checks = {
        "ours_beats_both_baselines_every_seed_and_k": all(
            statistics.fmean(raw["table1"][str(seed)]["ours"][str(k)])
            < statistics.fmean(raw["table1"][str(seed)][baseline][str(k)])
            for seed in SEEDS
            for k in KS
            for baseline in ("mogfn", "hngfn")
        ),
        "trained_baselines_within_0_02_of_paper": all(
            abs(recomputed[method][str(k)]["mean"] - paper[method][str(k)]) <= 0.02
            for method in ("mogfn", "hngfn")
            for k in KS
        ),
        "paper_ours_point_is_0_003_each_k": all(
            paper["ours"][str(k)] == PAPER_OURS for k in KS
        ),
        "paper_ours_point_below_every_95pct_ci": all(
            recomputed["ours"][str(k)]["ci95_t"][0] > PAPER_OURS for k in KS
        ),
        "observed_ours_means_materially_above_point": all(
            recomputed["ours"][str(k)]["mean"] > 0.005 for k in KS
        ),
    }
    metrics = {
        "paper_ours": {str(k): paper["ours"][str(k)] for k in KS},
        "observed_ours": {
            str(k): {
                "mean": recomputed["ours"][str(k)]["mean"],
                "ci95_t": recomputed["ours"][str(k)]["ci95_t"],
            }
            for k in KS
        },
        "qualitative_ordering_verified": checks[
            "ours_beats_both_baselines_every_seed_and_k"
        ],
        "rounding_limitation": (
            "The contract adjudicates the exact printed point 0.003. "
            "A three-decimal rounding interval would overlap the k=4 CI."
        ),
    }
    return checks, metrics


def claim_5_checks(raw: dict, recomputed: dict) -> tuple[dict, dict]:
    paper = raw["summary"]["paper_table"]["ensemble"]
    paired = {
        str(seed): {
            str(k): (
                statistics.fmean(raw["table1"][str(seed)]["ensemble"][str(k)])
                - statistics.fmean(raw["table1"][str(seed)]["ours"][str(k)])
            )
            for k in KS
        }
        for seed in SEEDS
    }
    checks = {
        "paired_improvement_at_least_0_05_every_seed_and_k": all(
            paired[str(seed)][str(k)] >= 0.05 for seed in SEEDS for k in KS
        ),
        "ensemble_aggregates_within_0_03_of_paper": all(
            abs(recomputed["ensemble"][str(k)]["mean"] - paper[str(k)]) <= 0.03
            for k in KS
        ),
    }
    metrics = {
        "paired_ensemble_minus_ours": paired,
        "observed_ensemble": {
            str(k): recomputed["ensemble"][str(k)]["mean"] for k in KS
        },
        "paper_ensemble": paper,
    }
    return checks, metrics


def claim_6_checks(raw: dict) -> tuple[dict, dict]:
    rows = raw["distortion"]
    primary = [row for row in rows if row["custom_dist"] in PRIMARY_FIGURE_5]
    high_better = [
        row["high_g_median_relative_deviation"]
        < row["low_g_median_relative_deviation"]
        for row in rows
    ]
    negative_spearman = [
        row["spearman_g_vs_relative_deviation"] < 0 for row in rows
    ]
    checks = {
        "36_appendix_stress_audits": len(rows) == 36,
        "all_12_operators_each_seed": (
            len({row["custom_dist"] for row in rows}) == 12
            and all(
                sorted(
                    row["seed"]
                    for row in rows
                    if row["custom_dist"] == operator
                )
                == SEEDS
                for operator in {row["custom_dist"] for row in rows}
            )
        ),
        "mass_state_and_identity_sanity": all(
            row["states"] == 1024
            and abs(row["induced_mass"] - 1.0) < 1e-5
            and abs(row["target_mass"] - 1.0) < 1e-12
            and row["delta_identity_max_abs_residual"] < 1e-4
            for row in rows
        ),
        "six_exact_primary_figure_5_rows": (
            len(primary) == 6
            and {row["custom_dist"] for row in primary} == PRIMARY_FIGURE_5
            and sorted({row["seed"] for row in primary}) == SEEDS
        ),
        "primary_high_g_closer_than_low_g_every_seed_and_operator": all(
            row["high_g_median_relative_deviation"]
            < row["low_g_median_relative_deviation"]
            for row in primary
        ),
        "primary_high_g_error_share_below_target_mass_every_seed_and_operator": all(
            row["high_g_l1_share"] < row["high_g_target_mass"] for row in primary
        ),
    }
    metrics = {
        "primary_rows": [
            {
                key: row[key]
                for key in (
                    "seed",
                    "custom_dist",
                    "high_g_median_relative_deviation",
                    "low_g_median_relative_deviation",
                    "high_g_target_mass",
                    "high_g_l1_share",
                    "outlier_count",
                    "outlier_target_mass",
                    "outliers_in_low_half_fraction",
                    "spearman_g_vs_relative_deviation",
                )
            }
            for row in primary
        ],
        "appendix_stress": {
            "high_g_closer_count": sum(high_better),
            "total": len(rows),
            "negative_spearman_count": sum(negative_spearman),
            "median_high_g_relative_deviation": statistics.median(
                row["high_g_median_relative_deviation"] for row in rows
            ),
            "median_low_g_relative_deviation": statistics.median(
                row["low_g_median_relative_deviation"] for row in rows
            ),
            "limitation": (
                "Five of 36 broader appendix seed/operator audits reverse the "
                "high-vs-low ordering; the VERIFIED verdict is scoped to the "
                "paper's primary Figure 5 compositions."
            ),
        },
    }
    return checks, metrics


def apply_negative_control(raw: dict, claim: int) -> None:
    if claim == 2:
        for seed in SEEDS:
            for k in KS:
                raw["table1"][str(seed)]["ours"][str(k)] = [PAPER_OURS] * 128
    elif claim == 5:
        for seed in SEEDS:
            for k in KS:
                raw["table1"][str(seed)]["ensemble"][str(k)] = list(
                    raw["table1"][str(seed)]["ours"][str(k)]
                )
    elif claim == 6:
        for row in raw["distortion"]:
            if row["custom_dist"] in PRIMARY_FIGURE_5:
                (
                    row["high_g_median_relative_deviation"],
                    row["low_g_median_relative_deviation"],
                ) = (
                    row["low_g_median_relative_deviation"],
                    row["high_g_median_relative_deviation"],
                )


def verify(path: Path, claim: int, negative_control: bool) -> dict:
    input_sha256 = sha256(path)
    raw = json.loads(path.read_text())
    if negative_control:
        apply_negative_control(raw, claim)
    common, recomputed = common_checks(raw, input_sha256)
    if claim == 2:
        specific, metrics = claim_2_checks(raw, recomputed)
        verdict = "FALSIFIED"
    elif claim == 5:
        specific, metrics = claim_5_checks(raw, recomputed)
        verdict = "VERIFIED"
    else:
        specific, metrics = claim_6_checks(raw)
        verdict = "VERIFIED"
    checks = {**common, **specific}
    return {
        "claim": claim,
        "verdict": verdict,
        "negative_control": negative_control,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "input_sha256": input_sha256,
        "expected_sha256": EXPECTED_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", type=int, choices=(2, 5, 6), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    result = verify(args.input, args.claim, args.negative_control)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
