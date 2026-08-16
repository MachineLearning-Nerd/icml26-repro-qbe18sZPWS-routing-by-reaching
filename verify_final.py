#!/usr/bin/env python3
"""Fail-closed checks for the published repository surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_REPOSITORY = "MachineLearning-Nerd/icml26-routing-by-reaching-audit"
CANONICAL_NAME = "MachineLearning-Nerd"
CANONICAL_EMAIL = "MachineLearning-Nerd@users.noreply.github.com"
EXPECTED_BRANCHES = {
    "main",
    "audit/molecule-claim-prerequisites",
    "audit/qm9-hdf-compatibility",
    "evidence/cumulative-claim-contracts",
    "baseline/frozen-4-12",
    "evidence/full-seeded-hypergrid",
    "experiment/hf-cpu-threaded-hypergrid",
    "experiment/hypergrid-seed-1337",
    "experiment/hypergrid-seed-20260719",
    "release/illustrated-report",
    "experiment/local-cpu-threaded-hypergrid",
    "audit/profile-hypergrid-cpu",
    "audit/profile-single-thread",
    "release/prepublication-report",
    "release/published-space-mirror",
}
EXPECTED_CLAIMS = {
    "C1": "VERIFIED_SCOPED",
    "C2": "FALSIFIED_EXACT_POINT",
    "C3": "BLOCKED",
    "C4": "BLOCKED",
    "C5": "VERIFIED_SCOPED",
    "C6": "VERIFIED_PRIMARY_SCOPE",
}
EXPECTED_HASHES = {
    "evidence/full_grid_raw.json": "d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec",
    "repro/evidence/full_grid_raw.json": "d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec",
    "evidence/claim-6/direct_state_audit.json": "dfe60d7381035df45239aab16d266ae87e2a82625fc2163fac54f43df57c28ef",
}
REQUIRED_FILES = {
    "README.md",
    "STATUS.md",
    "REPORT.md",
    "CLAIM_EVIDENCE.md",
    "BRANCH_AUDIT.md",
    "docs/SOURCE_AUDIT.md",
    "ENVIRONMENT.md",
    "AUTHOR_THANK_YOU.md",
    "CITATION.cff",
    "claims.json",
    "EVIDENCE_MANIFEST.json",
    "verify_final.py",
}
REQUIRED_EVIDENCE_PATHS = {
    "evidence/verifier_output_v3.json",
    "evidence/campaign_summary_v3.json",
    "evidence/release_comparison_v5.json",
    "evidence/claim-1",
    "evidence/claim-2/independent_checker_v3.json",
    "evidence/claim-3/molecule_prerequisite_audit.json",
    "evidence/claim-4/molecule_prerequisite_audit.json",
    "evidence/claim-5/independent_checker.json",
    "evidence/claim-6/direct_state_check.json",
    "repro/evidence/full_grid_raw.json",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def run(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def read_json(relative_path: str) -> object:
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_branches() -> set[str]:
    prefix = "refs/remotes/origin/"
    refs = run("git", "for-each-ref", "refs/remotes/origin", "--format=%(refname)")
    return {
        ref.strip()[len(prefix) :]
        for ref in refs.splitlines()
        if ref.strip().startswith(prefix) and ref.strip() != prefix + "HEAD"
    }


def local_branches() -> set[str]:
    refs = run("git", "for-each-ref", "refs/heads", "--format=%(refname:strip=2)")
    return {ref.strip() for ref in refs.splitlines() if ref.strip()}


def verify_remote() -> None:
    remote = run("git", "config", "--get", "remote.origin.url").strip()
    normalized = remote.removesuffix(".git")
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    if normalized.endswith(EXPECTED_REPOSITORY):
        return
    fail(f"origin is {remote!r}, expected {EXPECTED_REPOSITORY!r}")


def verify_history() -> None:
    records = run(
        "git",
        "log",
        "--all",
        "--format=%an%x00%ae%x00%cn%x00%ce",
    ).splitlines()
    if not records:
        fail("no reachable commits")
    expected = f"{CANONICAL_NAME}\x00{CANONICAL_EMAIL}\x00{CANONICAL_NAME}\x00{CANONICAL_EMAIL}"
    unexpected = sorted({record for record in records if record != expected})
    if unexpected:
        fail(f"non-canonical reachable identities: {unexpected}")
    messages = run("git", "log", "--all", "--format=%B")
    if "Co-authored-by:" in messages:
        fail("co-author trailer found")


def main() -> int:
    missing = sorted(path for path in REQUIRED_FILES if not (ROOT / path).is_file())
    if missing:
        fail(f"missing required audit files: {missing}")
    missing_evidence = sorted(path for path in REQUIRED_EVIDENCE_PATHS if not (ROOT / path).exists())
    if missing_evidence:
        fail(f"missing required evidence paths: {missing_evidence}")

    claims = read_json("claims.json")
    manifest = read_json("EVIDENCE_MANIFEST.json")
    if not isinstance(claims, dict) or not isinstance(manifest, dict):
        fail("claims and manifest must be JSON objects")
    if claims.get("repository") != EXPECTED_REPOSITORY:
        fail("claims repository marker is wrong")
    if manifest.get("repository") != EXPECTED_REPOSITORY:
        fail("manifest repository marker is wrong")
    if claims.get("overall_status") != "MIXED_RESULTS_WITH_LIMITATIONS":
        fail("overall status is wrong")
    if manifest.get("claim_statuses") != EXPECTED_CLAIMS:
        fail("manifest claim statuses are wrong")
    claim_rows = claims.get("claims")
    if not isinstance(claim_rows, list):
        fail("claims must contain a list")
    observed = {row.get("id"): row.get("status") for row in claim_rows if isinstance(row, dict)}
    if observed != EXPECTED_CLAIMS:
        fail(f"claim ledger statuses are wrong: {observed}")

    for relative_path, expected_hash in EXPECTED_HASHES.items():
        actual_hash = sha256(relative_path)
        if actual_hash != expected_hash:
            fail(f"hash mismatch for {relative_path}: {actual_hash}")
    manifest_hashes = {
        item.get("path"): item.get("sha256")
        for item in manifest.get("content_addressed_evidence", [])
        if isinstance(item, dict)
    }
    if any(manifest_hashes.get(path) != expected for path, expected in EXPECTED_HASHES.items()):
        fail("manifest content-addressed hashes do not match")

    verifier = read_json("evidence/verifier_output_v3.json")
    if not isinstance(verifier, dict) or verifier.get("passed") is not True:
        fail("recorded cumulative verifier is not passed")
    if manifest.get("branches", {}).get("expected_final") is None:
        fail("manifest has no final branch inventory")
    if set(manifest["branches"]["expected_final"]) != EXPECTED_BRANCHES:
        fail("manifest final branch inventory is wrong")

    verify_remote()
    observed_remote = remote_branches()
    if observed_remote != EXPECTED_BRANCHES:
        fail(f"remote branches differ: {sorted(observed_remote)}")
    observed_local = local_branches()
    if observed_local not in ({"main"}, EXPECTED_BRANCHES):
        fail(f"local branches are unexpected: {sorted(observed_local)}")
    verify_history()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for marker in ("claims.json", "CITATION.cff", "AUTHOR_THANK_YOU.md", "MIXED_RESULTS_WITH_LIMITATIONS"):
        if marker not in readme:
            fail(f"README is missing marker {marker!r}")
    print("PASS: final repository audit, evidence hashes, branch inventory, remote, and attribution")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, subprocess.CalledProcessError, OSError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
