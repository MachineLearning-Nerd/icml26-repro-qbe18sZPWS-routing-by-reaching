#!/usr/bin/env python3
"""Independent verifier for direct state-level Claim 6 evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
from pathlib import Path


SEEDS = [604, 1337, 20260719]
OPERATORS = ["harmonic_mean_circle12", "contrast_circle12"]
EXPECTED_ROWS = {(seed, operator) for seed in SEEDS for operator in OPERATORS}


def region_stats(indices, g, induced, delta, reference):
    normalized = [delta[index] / reference for index in indices]
    target = [value / sum(g) for value in g]
    l1 = [abs(observed - expected) for observed, expected in zip(induced, target)]
    mean = statistics.fmean(normalized)
    variance = statistics.pvariance(normalized)
    return {
        "count": len(indices),
        "g_min": min(g[index] for index in indices),
        "g_max": max(g[index] for index in indices),
        "normalized_delta_mean": mean,
        "normalized_delta_variance": variance,
        "normalized_delta_std": math.sqrt(variance),
        "normalized_delta_cv": math.sqrt(variance) / max(abs(mean), 1e-300),
        "median_abs_relative_deviation": statistics.median(
            abs(value - 1.0) for value in normalized
        ),
        "rmse_relative_deviation": math.sqrt(
            statistics.fmean((value - 1.0) ** 2 for value in normalized)
        ),
        "target_mass": sum(target[index] for index in indices),
        "l1_error_share": (
            sum(l1[index] for index in indices) / max(sum(l1), 1e-300)
        ),
    }


def close(left, right, tolerance=1e-11):
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def summaries_match(recorded, recomputed):
    return all(
        recorded[key] == value
        if isinstance(value, int)
        else close(recorded[key], value)
        for key, value in recomputed.items()
    )


def verify(path: Path, negative_control: bool) -> dict:
    raw = json.loads(path.read_text())
    rows = copy.deepcopy(raw["rows"])
    if negative_control:
        for position, row in enumerate(rows):
            shuffled = list(row["state_values"]["n_m"])
            random.Random(912_000 + position).shuffle(shuffled)
            row["state_values"]["n_m"] = shuffled

    row_checks = []
    metrics = []
    for row in rows:
        values = row["state_values"]
        g = values["g"]
        u_m = values["u_m"]
        n_m = values["n_m"]
        recorded_delta = values["delta"]
        induced = values["induced_probability"]
        delta = [u / max(n, 1e-300) for u, n in zip(u_m, n_m)]
        reference = 1.0 / sum(g)
        order = sorted(range(len(g)), key=lambda index: (g[index], index))
        low = order[: len(order) // 2]
        high = order[-max(1, len(order) // 10) :]
        all_states = list(range(len(g)))
        summaries = {
            "all_states": region_stats(
                all_states, g, induced, delta, reference
            ),
            "bottom_half_g": region_stats(low, g, induced, delta, reference),
            "high_g_decile": region_stats(high, g, induced, delta, reference),
        }
        target = [value / sum(g) for value in g]
        equation_9 = [abs(value - reference) * weight for value, weight in zip(delta, g)]
        direct_l1 = [abs(observed - expected) for observed, expected in zip(induced, target)]
        delta_identity_residual = max(
            abs(value - observed / max(weight, 1e-300))
            for value, observed, weight in zip(delta, induced, g)
        )
        delta_record_residual = max(
            abs(value - recorded)
            for value, recorded in zip(delta, recorded_delta)
        )
        equation_9_residual = max(
            abs(left - right) for left, right in zip(equation_9, direct_l1)
        )
        high_stats = summaries["high_g_decile"]
        low_stats = summaries["bottom_half_g"]
        checks = {
            "all_1024_states": (
                len(g)
                == len(u_m)
                == len(n_m)
                == len(recorded_delta)
                == len(induced)
                == 1024
            ),
            "positive_g_and_n_m": min(g) > 0 and min(n_m) > 0,
            "recorded_delta_recomputes_from_u_m_over_n_m": (
                delta_record_residual < 1e-12
            ),
            "delta_matches_induced_probability_over_g": (
                delta_identity_residual < 1e-4
            ),
            "equation_9_recomputes_l1_contributions": equation_9_residual < 1e-8,
            "recorded_region_summaries_recompute": all(
                summaries_match(row["regions"][name], summary)
                for name, summary in summaries.items()
            ),
            "high_g_variance_below_bottom_half": (
                high_stats["normalized_delta_variance"]
                < low_stats["normalized_delta_variance"]
            ),
            "high_g_median_deviation_below_bottom_half": (
                high_stats["median_abs_relative_deviation"]
                < low_stats["median_abs_relative_deviation"]
            ),
            "high_g_rmse_below_bottom_half": (
                high_stats["rmse_relative_deviation"]
                < low_stats["rmse_relative_deviation"]
            ),
            "high_g_l1_share_below_target_mass": (
                high_stats["l1_error_share"] < high_stats["target_mass"]
            ),
        }
        row_checks.append(checks)
        metrics.append(
            {
                "seed": row["seed"],
                "operator": row["operator"],
                "high_g": high_stats,
                "bottom_half_g": low_stats,
                "delta_record_max_abs_residual": delta_record_residual,
                "delta_identity_max_abs_residual": delta_identity_residual,
                "equation_9_max_abs_residual": equation_9_residual,
            }
        )

    checks = {
        "paper_v1_contract": raw["paper"] == "arXiv:2602.21565v1",
        "claim_6": raw["claim"] == 6,
        "paper_scale_32x32": raw["grid"] == "32x32",
        "paper_training_20000_steps": raw["training_iterations"] == 20_000,
        "three_fixed_seeds": raw["seeds"] == SEEDS,
        "two_primary_operators": raw["operators"] == OPERATORS,
        "six_exact_rows": (
            len(rows) == 6
            and {(row["seed"], row["operator"]) for row in rows} == EXPECTED_ROWS
        ),
        "all_state_level_math_checks": all(
            check["all_1024_states"]
            and check["positive_g_and_n_m"]
            and check["recorded_delta_recomputes_from_u_m_over_n_m"]
            and check["delta_matches_induced_probability_over_g"]
            and check["equation_9_recomputes_l1_contributions"]
            and check["recorded_region_summaries_recompute"]
            for check in row_checks
        ),
        "high_g_variance_lower_in_all_rows": all(
            check["high_g_variance_below_bottom_half"] for check in row_checks
        ),
        "high_g_median_deviation_lower_in_all_rows": all(
            check["high_g_median_deviation_below_bottom_half"]
            for check in row_checks
        ),
        "high_g_rmse_lower_in_all_rows": all(
            check["high_g_rmse_below_bottom_half"] for check in row_checks
        ),
        "high_g_error_share_below_mass_in_all_rows": all(
            check["high_g_l1_share_below_target_mass"] for check in row_checks
        ),
    }
    passed = all(checks.values())
    return {
        "claim": 6,
        "negative_control": negative_control,
        "verdict": "VERIFIED" if passed else "REJECTED",
        "passed": passed,
        "checks": checks,
        "row_checks": row_checks,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evidence/claim-6/direct_state_audit.json"),
    )
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input, args.negative_control)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
