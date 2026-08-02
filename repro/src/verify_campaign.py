#!/usr/bin/env python3
"""Fail-closed verifier for the cumulative reproduction campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_FULL_GRID_SHA256 = (
    "d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec"
)
EXPECTED_DIRECT_CLAIM_6_SHA256 = (
    "dfe60d7381035df45239aab16d266ae87e2a82625fc2163fac54f43df57c28ef"
)
EXPECTED_VERDICTS = {
    "1": "VERIFIED",
    "2": "FALSIFIED",
    "3": "BLOCKED",
    "4": "BLOCKED",
    "5": "VERIFIED",
    "6": "VERIFIED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    summary = json.loads((root / "campaign_summary.json").read_text())
    exact = json.loads((root / "claim-1" / "raw_exact_summary.json").read_text())
    verdicts = {str(row["claim"]): row["verdict"] for row in summary["claims"]}
    molecule = json.loads(
        (root / "claim-3" / "molecule_prerequisite_audit.json").read_text()
    )
    molecule_checker = json.loads(
        (root / "claim-3" / "molecule_audit_checker.json").read_text()
    )
    molecule_negative = json.loads(
        (root / "claim-3" / "molecule_negative_control.json").read_text()
    )
    claim_checkers = {
        str(claim): json.loads(
            (root / f"claim-{claim}" / "independent_checker.json").read_text()
        )
        for claim in (2, 5, 6)
    }
    claim_negatives = {
        str(claim): json.loads(
            (root / f"claim-{claim}" / "negative_control.json").read_text()
        )
        for claim in (2, 5, 6)
    }
    direct_claim_6_path = root / "claim-6" / "direct_state_audit.json"
    direct_claim_6 = json.loads(direct_claim_6_path.read_text())
    direct_claim_6_checker = json.loads(
        (root / "claim-6" / "direct_state_check.json").read_text()
    )
    direct_claim_6_negative = json.loads(
        (root / "claim-6" / "direct_state_negative_control.json").read_text()
    )
    full_grid_paths = [
        root / f"claim-{claim}" / "full_grid_raw.json" for claim in (2, 5, 6)
    ]

    checks = {
        "all_six_claims_present": sorted(verdicts) == list("123456"),
        "exact_cumulative_verdicts": verdicts == EXPECTED_VERDICTS,
        "claim_1_all_512_settings": len(exact["claim_2"]["settings"]) == 512,
        "claim_1_max_l1": exact["claim_2"]["max_l1"] < 1e-12,
        "claim_1_flow_certificate": (
            exact["claim_2"]["max_flow_certificate_residual"] < 1e-11
        ),
        "negative_omit_reaching_rejected": (
            exact["negative_controls"]["omit_reaching_l1"] > 1e-2
        ),
        "negative_omit_partition_rejected": (
            exact["negative_controls"]["omit_partition_l1"] > 1e-3
        ),
        "negative_wrong_beta_rejected": (
            exact["negative_controls"]["apply_beta2_to_beta1_target_l1"] > 1e-3
        ),
        "no_inconclusive_or_toy_labels": all(
            value in {"VERIFIED", "FALSIFIED", "BLOCKED"}
            for value in verdicts.values()
        ),
        "claim_specific_checkers_passed": all(
            checker["passed"] and not checker["negative_control"]
            for checker in claim_checkers.values()
        ),
        "claim_specific_checker_verdicts": all(
            claim_checkers[claim]["verdict"] == EXPECTED_VERDICTS[claim]
            for claim in claim_checkers
        ),
        "claim_specific_negative_controls_rejected": all(
            negative["negative_control"] and not negative["passed"]
            for negative in claim_negatives.values()
        ),
        "claim_2_qualitative_ordering_retained": claim_checkers["2"]["metrics"][
            "qualitative_ordering_verified"
        ],
        "claim_2_exact_point_excluded": claim_checkers["2"]["checks"][
            "paper_ours_point_below_every_95pct_ci"
        ],
        "claim_5_paired_ablation_passed": claim_checkers["5"]["checks"][
            "paired_improvement_at_least_0_05_every_seed_and_k"
        ],
        "claim_6_primary_scope_passed": (
            claim_checkers["6"]["checks"][
                "primary_high_g_closer_than_low_g_every_seed_and_operator"
            ]
            and claim_checkers["6"]["checks"][
                "primary_high_g_error_share_below_target_mass_every_seed_and_operator"
            ]
        ),
        "claim_6_direct_evidence_content_addressed": (
            sha256(direct_claim_6_path)
            == EXPECTED_DIRECT_CLAIM_6_SHA256
            == summary["claim_6_direct_state_sha256"]
        ),
        "claim_6_direct_six_rows": (
            direct_claim_6["claim"] == 6
            and len(direct_claim_6["rows"]) == 6
        ),
        "claim_6_direct_checker_passed": (
            direct_claim_6_checker["passed"]
            and not direct_claim_6_checker["negative_control"]
        ),
        "claim_6_direct_negative_control_rejected": (
            direct_claim_6_negative["negative_control"]
            and not direct_claim_6_negative["passed"]
        ),
        "claim_6_direct_high_g_contract_passed": all(
            direct_claim_6_checker["checks"][name]
            for name in (
                "high_g_variance_lower_in_all_rows",
                "high_g_median_deviation_lower_in_all_rows",
                "high_g_rmse_lower_in_all_rows",
                "high_g_error_share_below_mass_in_all_rows",
            )
        ),
        "full_grid_raw_copies_content_addressed": all(
            sha256(path) == EXPECTED_FULL_GRID_SHA256 for path in full_grid_paths
        ),
        "molecule_audit_checker_passed": molecule_checker["passed"],
        "molecule_negative_control_rejected": molecule_negative[
            "validator_rejected"
        ],
        "molecule_checkpoints_missing": bool(
            molecule["checkpoint_inventory"]["missing"]
        ),
        "molecule_comparator_surfaces_missing": bool(
            molecule["source_surface_inventory"]["missing"]
        ),
        "molecule_threshold_drift_recorded": (
            not molecule["threshold_inventory"]["matches_paper_v1"]
        ),
    }
    result = {"passed": all(checks.values()), "checks": checks}
    (root / "verifier_output.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(".openresearch/artifacts"),
    )
    outcome = verify(parser.parse_args().artifacts)
    sys.exit(0 if outcome["passed"] else 1)
