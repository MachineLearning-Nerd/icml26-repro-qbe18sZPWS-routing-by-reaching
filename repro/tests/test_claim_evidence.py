from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from verify_claim_evidence import EXPECTED_SHA256, sha256, verify  # noqa: E402


RAW = ROOT / "repro" / "evidence" / "full_grid_raw.json"


def test_archived_parent_evidence_is_content_addressed() -> None:
    assert sha256(RAW) == EXPECTED_SHA256


def test_all_claim_specific_contracts_pass() -> None:
    for claim, verdict in ((2, "FALSIFIED"), (5, "VERIFIED"), (6, "VERIFIED")):
        result = verify(RAW, claim, negative_control=False)
        assert result["passed"]
        assert result["verdict"] == verdict


def test_all_claim_specific_negative_controls_fail() -> None:
    for claim in (2, 5, 6):
        result = verify(RAW, claim, negative_control=True)
        assert not result["passed"]
