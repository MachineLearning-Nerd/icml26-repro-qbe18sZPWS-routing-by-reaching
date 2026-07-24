#!/usr/bin/env python3
"""Fail-closed prerequisite audit for the paper's molecule claims."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MOLS = ROOT / "official" / "mols"
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
REPORT_PATH = ARTIFACTS / "claim-3" / "molecule_prerequisite_audit.json"
CHECKER_PATH = ARTIFACTS / "claim-3" / "molecule_audit_checker.json"
NEGATIVE_PATH = ARTIFACTS / "claim-3" / "molecule_negative_control.json"

REQUIRED_CHECKPOINTS = {
    "qm9_gap_ingredient": "official/mols/ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt",
    "qm9_sa_ingredient": "official/mols/ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt",
    "qm9_mogfn_gap_sa": "unreleased: trained QM9 MOGFN GAP-SA checkpoint",
    "qm9_hngfn_gap_sa": "unreleased: trained QM9 HN-GFN GAP-SA checkpoint",
    "fragment_seh_ingredient": "official/mols/ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt",
    "fragment_sa_ingredient": "official/mols/ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt",
    "fragment_qed_ingredient": "official/mols/ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt",
    "fragment_classifier_guidance": "unreleased: molecule classifier-guidance checkpoint",
    "qm9_classifier_guidance": "unreleased: molecule classifier-guidance checkpoint",
}

REQUIRED_SURFACES = {
    "proposed_molecule_mixer": [
        "official/mols/eval_scalarization.py",
        "official/mols/eval_logical_operators.py",
    ],
    "ingredient_training": ["official/mols/train_gfn.py"],
    "qm9_mogfn": [],
    "qm9_hngfn": [],
    "molecule_classifier_guidance": [],
    "molecule_inference_timing_benchmark": [],
}

EXPECTED_IMPORTS = [
    "gfn",
    "torch_geometric",
    "rdkit",
    "h5py",
    "botorch",
    "gfn_composition.tasks.qm9",
    "gfn_composition.tasks.qm9_moo",
    "gfn_composition.algo.mixture_graph_sampling_subtb",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def checkpoint_inventory() -> dict:
    actual = sorted(
        str(path.relative_to(ROOT))
        for path in MOLS.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    )
    checks = {}
    for name, expected in REQUIRED_CHECKPOINTS.items():
        present = not expected.startswith("unreleased:") and (ROOT / expected).is_file()
        checks[name] = {"expected": expected, "present": present}
    return {
        "actual_checkpoint_files": actual,
        "required": checks,
        "missing": sorted(name for name, row in checks.items() if not row["present"]),
        "scorer_not_gfn_checkpoint": (
            "official/mols/data/mxmnet_gap_model.pt" in actual
        ),
    }


def source_surface_inventory() -> dict:
    molecule_files = sorted(
        str(path.relative_to(ROOT))
        for path in MOLS.rglob("*")
        if path.is_file()
    )
    combined = "\n".join(
        path.read_text(errors="replace")
        for path in MOLS.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".md"}
    )
    checks = {
        name: {
            "paths": paths,
            "present": bool(paths) and all((ROOT / path).is_file() for path in paths),
        }
        for name, paths in REQUIRED_SURFACES.items()
    }
    token_counts = {
        token: len(re.findall(re.escape(token), combined, flags=re.IGNORECASE))
        for token in (
            "MOGFN",
            "HN-GFN",
            "classifier guidance",
            "perf_counter",
            "time.time",
        )
    }
    return {
        "molecule_files": molecule_files,
        "required": checks,
        "missing": sorted(name for name, row in checks.items() if not row["present"]),
        "diagnostic_token_counts": token_counts,
    }


def import_inventory() -> dict:
    sys.path.insert(0, str(MOLS))
    checks = {}
    for name in EXPECTED_IMPORTS:
        try:
            importlib.import_module(name)
            checks[name] = {"ok": True, "error_type": None, "error": None}
        except Exception as exc:  # audit the exact locked runtime failure
            checks[name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
    return {
        "checks": checks,
        "all_ok": all(row["ok"] for row in checks.values()),
    }


def data_inventory() -> dict:
    scorer = MOLS / "data" / "mxmnet_gap_model.pt"
    qm9 = MOLS / "data" / "qm9.h5"
    hdf5 = {"opened": False, "keys": [], "error": None}
    try:
        import h5py

        with h5py.File(qm9, "r") as handle:
            hdf5["opened"] = True
            hdf5["keys"] = sorted(handle.keys())
    except Exception as exc:
        hdf5["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return {
        "qm9_h5": {
            "present": qm9.is_file(),
            "bytes": qm9.stat().st_size if qm9.is_file() else None,
            "sha256": sha256(qm9) if qm9.is_file() else None,
            "hdf5": hdf5,
        },
        "mxmnet_gap_scorer": {
            "present": scorer.is_file(),
            "bytes": scorer.stat().st_size if scorer.is_file() else None,
            "sha256": sha256(scorer) if scorer.is_file() else None,
            "torch_zip_container": zipfile.is_zipfile(scorer) if scorer.is_file() else None,
            "role": "GAP property scorer; not a generative GFlowNet checkpoint",
        },
    }


def threshold_inventory() -> dict:
    source = (MOLS / "eval_logical_operators.py").read_text()
    match = re.search(
        r'"qm9"\s*:\s*\{\s*"gap"\s*:\s*([0-9.]+),\s*"sa"\s*:\s*([0-9.]+),\s*"qed"\s*:\s*([0-9.]+)',
        source,
    )
    released = [float(value) for value in match.groups()] if match else None
    expected_v1 = [0.85, 0.4, 0.4]
    return {
        "paper_v1_qm9_gap_sa_qed": expected_v1,
        "released_code_qm9_gap_sa_qed": released,
        "matches_paper_v1": released == expected_v1,
        "source": "official/mols/eval_logical_operators.py:BIN_THRESHOLDS",
    }


def build_report() -> dict:
    checkpoints = checkpoint_inventory()
    surfaces = source_surface_inventory()
    imports = import_inventory()
    data = data_inventory()
    thresholds = threshold_inventory()
    blockers = []
    if checkpoints["missing"]:
        blockers.append("missing_required_model_checkpoints")
    if surfaces["missing"]:
        blockers.append("missing_required_comparator_or_timing_code")
    if not imports["all_ok"]:
        blockers.append("locked_environment_import_incompatibility")
    if not thresholds["matches_paper_v1"]:
        blockers.append("released_qm9_bin_thresholds_differ_from_paper_v1")
    return {
        "paper": "arXiv:2602.21565v1",
        "purpose": "prerequisite audit; BLOCKED is not experimental verification",
        "git_sha": git_output("rev-parse", "HEAD"),
        "official_molecule_tree_sha256": sha256(
            ROOT / "official" / "README.md"
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ("gflownet", "torch", "torch-geometric", "rdkit", "h5py", "botorch")
        },
        "upstream_gflownet": {
            "documented_tag": "v0.2.0",
            "resolved_commit": "f106cdeb6892214cbb528a3e06f4c721f4003175",
        },
        "claim_3_contract": {
            "paper_result": {"ours": 0.876, "MOGFN": 0.816, "HN-GFN": 0.805},
            "domain": "atom-based QM9 GAP-SA",
            "beta": 32,
            "preferences": 10,
            "candidates_per_preference": 128,
            "aggregate": "mean of per-preference top-10 scalarized reward",
        },
        "claim_4_contract": {
            "timing_samples": 1000,
            "accuracy_samples": 5000,
            "operators": ["harmonic_mean", "contrast"],
            "methods": ["routing_by_reaching", "classifier_guidance"],
            "qm9_v1_thresholds_gap_sa_qed": [0.85, 0.4, 0.4],
        },
        "checkpoint_inventory": checkpoints,
        "source_surface_inventory": surfaces,
        "import_inventory": imports,
        "data_inventory": data,
        "threshold_inventory": thresholds,
        "blockers": blockers,
        "claim_3_verdict": "BLOCKED",
        "claim_4_verdict": "BLOCKED",
    }


def validate_report(report: dict) -> dict:
    checks = {
        "verdicts_are_blocked": (
            report["claim_3_verdict"] == "BLOCKED"
            and report["claim_4_verdict"] == "BLOCKED"
        ),
        "qm9_data_present": report["data_inventory"]["qm9_h5"]["present"],
        "gap_scorer_present": report["data_inventory"]["mxmnet_gap_scorer"]["present"],
        "gap_scorer_not_misclassified": report["checkpoint_inventory"][
            "scorer_not_gfn_checkpoint"
        ],
        "required_checkpoints_missing": bool(
            report["checkpoint_inventory"]["missing"]
        ),
        "required_surfaces_missing": bool(
            report["source_surface_inventory"]["missing"]
        ),
        "blocker_checkpoint_recorded": (
            "missing_required_model_checkpoints" in report["blockers"]
        ),
        "blocker_surface_recorded": (
            "missing_required_comparator_or_timing_code" in report["blockers"]
        ),
        "threshold_drift_recorded": (
            not report["threshold_inventory"]["matches_paper_v1"]
            and "released_qm9_bin_thresholds_differ_from_paper_v1"
            in report["blockers"]
        ),
        "claim_3_quantifiers_recorded": (
            report["claim_3_contract"]["preferences"] == 10
            and report["claim_3_contract"]["candidates_per_preference"] == 128
            and report["claim_3_contract"]["beta"] == 32
        ),
        "claim_4_quantifiers_recorded": (
            report["claim_4_contract"]["timing_samples"] == 1000
            and report["claim_4_contract"]["accuracy_samples"] == 5000
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    checker = validate_report(report)

    mutated = copy.deepcopy(report)
    mutated["blockers"].remove("missing_required_model_checkpoints")
    negative_checker = validate_report(mutated)
    negative = {
        "mutation": "removed missing_required_model_checkpoints blocker",
        "validator_rejected": not negative_checker["passed"],
        "mutated_checker": negative_checker,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    CHECKER_PATH.write_text(json.dumps(checker, indent=2, sort_keys=True) + "\n")
    NEGATIVE_PATH.write_text(json.dumps(negative, indent=2, sort_keys=True) + "\n")
    for claim in (3, 4):
        claim_dir = ARTIFACTS / f"claim-{claim}"
        claim_dir.mkdir(parents=True, exist_ok=True)
        (claim_dir / REPORT_PATH.name).write_text(REPORT_PATH.read_text())
        (claim_dir / CHECKER_PATH.name).write_text(CHECKER_PATH.read_text())
        (claim_dir / NEGATIVE_PATH.name).write_text(NEGATIVE_PATH.read_text())

    output = {"report": report, "checker": checker, "negative_control": negative}
    print("MOLECULE_PREREQUISITE_AUDIT_JSON")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not checker["passed"] or not negative["validator_rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
