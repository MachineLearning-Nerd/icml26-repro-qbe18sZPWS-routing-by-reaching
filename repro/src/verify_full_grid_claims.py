#!/usr/bin/env python3
"""Independent fail-closed checker for full HyperGrid claim evidence."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path


SEEDS = [604, 1337, 20260719]


def verify(raw: dict, negative_control: bool) -> dict:
    candidate = copy.deepcopy(raw)
    if negative_control:
        for seed in SEEDS:
            for k in range(2, 6):
                candidate["table1"][str(seed)]["ours"][str(k)] = candidate[
                    "table1"
                ][str(seed)]["ensemble"][str(k)]

    recomputed_means = {}
    for method in ("ours", "ensemble", "mogfn", "hngfn"):
        recomputed_means[method] = {}
        for k in range(2, 6):
            recomputed_means[method][str(k)] = statistics.fmean(
                statistics.fmean(
                    candidate["table1"][str(seed)][method][str(k)]
                )
                for seed in SEEDS
            )

    checks = {
        "three_exact_seeds": sorted(map(int, candidate["table1"])) == SEEDS,
        "128_preferences_every_cell": all(
            len(candidate["table1"][str(seed)][method][str(k)]) == 128
            for seed in SEEDS
            for method in ("ours", "ensemble", "mogfn", "hngfn")
            for k in range(2, 6)
        ),
        "all_l1_in_valid_range": all(
            0 <= value <= 2
            for seed in SEEDS
            for method in ("ours", "ensemble", "mogfn", "hngfn")
            for k in range(2, 6)
            for value in candidate["table1"][str(seed)][method][str(k)]
        ),
        "summary_means_recompute": all(
            abs(
                recomputed_means[method][str(k)]
                - raw["summary"]["aggregates"][method][str(k)]["mean"]
            )
            < 1e-12
            for method in ("ours", "ensemble", "mogfn", "hngfn")
            for k in range(2, 6)
        ),
        "paired_reaching_improvement": all(
            statistics.fmean(
                candidate["table1"][str(seed)]["ensemble"][str(k)]
            )
            - statistics.fmean(
                candidate["table1"][str(seed)]["ours"][str(k)]
            )
            >= 0.05
            for seed in SEEDS
            for k in range(2, 6)
        ),
        "36_distortion_audits": len(candidate["distortion"]) == 36,
        "distortion_mass_sanity": all(
            row["states"] == 1024
            and abs(row["induced_mass"] - 1) < 1e-5
            and abs(row["target_mass"] - 1) < 1e-12
            and row["delta_identity_max_abs_residual"] < 1e-4
            for row in candidate["distortion"]
        ),
        "recorded_verdicts_are_fail_closed": all(
            candidate["summary"][f"claim_{claim}"]["verdict"]
            in {"VERIFIED", "BLOCKED"}
            for claim in (2, 5, 6)
        ),
    }
    return {
        "negative_control": negative_control,
        "passed": all(checks.values()),
        "checks": checks,
        "recomputed_means": recomputed_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    result = verify(json.loads(args.input.read_text()), args.negative_control)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
