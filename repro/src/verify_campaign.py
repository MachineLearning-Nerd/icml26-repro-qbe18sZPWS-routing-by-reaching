#!/usr/bin/env python3
"""Fail-closed verifier for the cumulative reproduction campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def verify(root: Path) -> dict:
    summary = json.loads((root / "campaign_summary.json").read_text())
    exact = json.loads((root / "claim-1" / "raw_exact_summary.json").read_text())
    verdicts = {str(row["claim"]): row["verdict"] for row in summary["claims"]}

    checks = {
        "claim_1_verdict_verified": verdicts["1"] == "VERIFIED",
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
