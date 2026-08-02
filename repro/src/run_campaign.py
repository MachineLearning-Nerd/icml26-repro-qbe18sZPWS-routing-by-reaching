#!/usr/bin/env python3
"""Fixed entrypoint for the claim-by-claim cumulative reproduction suite."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
FIXED_COMMAND = "uv sync --frozen && uv run python repro/src/run_campaign.py"
SEEDS = [604, 1337, 20260719]
ARCHIVED_FULL_GRID = ROOT / "repro" / "evidence" / "full_grid_raw.json"
ARCHIVED_FULL_GRID_SHA256 = (
    "d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec"
)
SOURCE_FULL_GRID_RUN = "fd92e2bc-ea94-4004-a820-62abf3e5e917"
SOURCE_FULL_GRID_COMMIT = "6c89eebcb076aab1076272d38a86529d649e2709"


CLAIMS = {
    1: {
        "paper_result": "Exact recovery for every admissible linear scalarization.",
        "anchor": "arXiv v1, Proposition 4.1, Section 4.2, Appendix A.1",
        "contract": (
            "On the shared 32x32 DAG at beta=1, for k=2..5 and 128 "
            "preferences per k, require max terminal L1 < 1e-12 and an "
            "independent flow-conservation residual < 1e-11."
        ),
    },
    2: {
        "paper_result": (
            "Table 1 L1: ours 0.003 for k=2..5; MOGFN 0.021-0.048; "
            "HN-GFN 0.017-0.035."
        ),
        "anchor": "arXiv v1, Sections 5.1-5.2, Table 1",
        "contract": (
            "Train the paper's neural ingredient GFNs, MOGFN, and HN-GFN on "
            "the 32x32 grid; evaluate exactly 128 simplex preferences for "
            "each k=2..5 over seeds 604, 1337, and 20260719. FALSIFY the "
            "exact printed ours=0.003 result only if 0.003 lies below the "
            "two-sided 95% t interval for every k. Independently require ours "
            "to beat both trained baselines in every seed/k cell and baseline "
            "aggregates to be within absolute L1 0.02 of the paper."
        ),
    },
    3: {
        "paper_result": "QM9 GAP-SA mean top-10 reward: ours 0.876, MOGFN 0.816, HN-GFN 0.805.",
        "anchor": "arXiv v1, Sections 6.1-6.2, Table 3",
        "contract": (
            "Using atom-based QM9 at beta=32, evaluate 10 evenly spaced "
            "GAP-SA preferences, 128 candidates each, and aggregate the "
            "per-preference top-10 scalarized reward for ours, MOGFN, and HN-GFN."
        ),
    },
    4: {
        "paper_result": (
            "Logical composition is 40-70x faster than classifier guidance "
            "and has comparable or higher target-bin accuracy."
        ),
        "anchor": "arXiv v1, Sections 6.2-6.3, Tables 4-5",
        "contract": (
            "On the official molecule sampler, time 1,000 logical-composition "
            "samples for both methods after warmup and evaluate target-bin "
            "accuracy on 5,000 samples for harmonic mean and contrast."
        ),
    },
    5: {
        "paper_result": "No-reaching ensemble L1 0.098-0.117 versus ours 0.003.",
        "anchor": "arXiv v1, Sections 5.1-5.2, Table 1",
        "contract": (
            "With the same trained neural ingredients and 512 grid settings "
            "as Claim 2, remove only u_i(s), report paired L1 and uncertainty, "
            "and require a paired improvement >=0.05 in every seed/k cell, "
            "with ensemble aggregates within absolute L1 0.03 of the paper."
        ),
    },
    6: {
        "paper_result": (
            "For nonlinear operators delta(x) stays near 1/Z_M in "
            "high-composition-value regions; deviations occur mainly at low G."
        ),
        "anchor": "arXiv v1, Section 5.3, Figure 5 and appendix panels",
        "contract": (
            "For the two primary Figure 5 nonlinear compositions "
            "(pCircle1 harmonic-mean pCircle2 and pCircle1 contrast pCircle2), "
            "directly compute delta, G, 1/Z_M, and quantile-binned relative "
            "deviation for three seeds. Require the high-G decile deviation "
            "to be below the bottom-half deviation and its L1-error share to "
            "be below its target-mass share in all six rows. Audit all 12 "
            "appendix settings as a broader, reported "
            "stress test without silently strengthening the primary claim."
        ),
    },
}


def run_checked(args: list[str], name: str) -> tuple[str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    print(f"\n===== {name} ({elapsed:.2f}s) =====")
    print(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{name} exited {proc.returncode}")
    return proc.stdout, elapsed


def run_expected_failure(args: list[str], name: str) -> tuple[str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    print(f"\n===== {name} ({elapsed:.2f}s; expected nonzero) =====")
    print(proc.stdout)
    if proc.returncode == 0:
        raise RuntimeError(f"{name} unexpectedly passed")
    return proc.stdout, elapsed


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def cumulative_verdicts(full_grid: dict) -> dict[int, tuple[str, str]]:
    aggregates = full_grid["summary"]["aggregates"]
    return {
        1: (
            "VERIFIED",
            "Retained full-scale exact theorem regression: 512/512 settings pass.",
        ),
        2: (
            "FALSIFIED",
            "The source-faithful neural run preserves the qualitative ordering "
            "against both baselines in all 12 seed/k cells, but its ours means "
            + ", ".join(
                f"k={k} {aggregates['ours'][str(k)]['mean']:.4f} "
                f"(95% CI {aggregates['ours'][str(k)]['ci95_t'][0]:.4f}-"
                f"{aggregates['ours'][str(k)]['ci95_t'][1]:.4f})"
                for k in range(2, 6)
            )
            + " exclude the exact printed 0.003 point.",
        ),
        3: (
            "BLOCKED",
            "The released tree has QM9 data and a GAP scorer, but lacks the "
            "QM9 ingredient, MOGFN, and HN-GFN checkpoints required by the "
            "exact 10-preference, 128-candidate GAP-SA comparison.",
        ),
        4: (
            "BLOCKED",
            "The released tree lacks molecule classifier-guidance code/checkpoints "
            "and a timing benchmark; its QM9 SA/QED bin thresholds are 0.3, "
            "whereas arXiv v1 Table 4 specifies 0.4.",
        ),
        5: (
            "VERIFIED",
            "The faithful no-reaching ablation is worse in every seed/k cell: "
            + ", ".join(
                f"k={k} ensemble={aggregates['ensemble'][str(k)]['mean']:.4f} "
                f"vs ours={aggregates['ours'][str(k)]['mean']:.4f}"
                for k in range(2, 6)
            ),
        ),
        6: (
            "VERIFIED",
            "All six source-primary Figure 5 seed/operator rows have lower "
            "relative distortion in the high-G decile than in the bottom "
            "half, and high-G L1-error share is below its target-mass share. "
            "The broader appendix stress "
            "audit has 31/36 rows in the same direction, so the verdict is "
            "not generalized beyond the primary scope.",
        ),
    }


def main() -> None:
    campaign_start = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    exact_path = ARTIFACTS / "claim-1" / "raw_exact_summary.json"
    exact_log, exact_seconds = run_checked(
        [
            sys.executable,
            "repro/src/run_routing_by_reaching.py",
            "--output",
            str(exact_path),
        ],
        "exact full-grid regression",
    )
    neural_log, neural_seconds = run_checked(
        [sys.executable, "repro/src/train_neural_gfns.py"],
        "cached neural-ingredient regression",
    )
    checker_log, checker_seconds = run_checked(
        [sys.executable, "-m", "pytest", "-q"],
        "independent pytest checker",
    )
    molecule_log, molecule_seconds = run_checked(
        [sys.executable, "repro/src/audit_molecule_claims.py"],
        "molecule prerequisite audit",
    )

    exact = json.loads(exact_path.read_text())
    neural_path = ROOT / "outputs" / "neural_composition.json"
    neural = json.loads(neural_path.read_text())
    full_grid_path = ARTIFACTS / "claim-2" / "full_grid_raw.json"
    if sha256(ARCHIVED_FULL_GRID) != ARCHIVED_FULL_GRID_SHA256:
        raise RuntimeError("archived full-grid evidence SHA-256 mismatch")
    full_grid_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARCHIVED_FULL_GRID, full_grid_path)
    full_grid = json.loads(full_grid_path.read_text())
    verdicts = cumulative_verdicts(full_grid)

    claim_checker_logs = {}
    claim_negative_logs = {}
    claim_checker_seconds = {}
    for claim_id in (2, 5, 6):
        claim_checker_logs[claim_id], claim_checker_seconds[str(claim_id)] = (
            run_checked(
                [
                    sys.executable,
                    "repro/src/verify_claim_evidence.py",
                    "--claim",
                    str(claim_id),
                    "--input",
                    str(full_grid_path),
                ],
                f"Claim {claim_id} independent evidence checker",
            )
        )
        claim_negative_logs[claim_id], negative_seconds = run_expected_failure(
            [
                sys.executable,
                "repro/src/verify_claim_evidence.py",
                "--claim",
                str(claim_id),
                "--input",
                str(full_grid_path),
                "--negative-control",
            ],
            f"Claim {claim_id} negative control",
        )
        claim_checker_seconds[f"{claim_id}_negative"] = negative_seconds
    report_figure_log, report_figure_seconds = run_checked(
        [
            sys.executable,
            "repro/src/build_report_figures.py",
            "--output",
            str(ARTIFACTS / "report" / "images"),
        ],
        "report figure regeneration",
    )
    (ARTIFACTS / "report" / "figure_runner_output.txt").write_text(
        report_figure_log
    )

    metadata = {
        "git_sha": git_sha(),
        "fixed_command": FIXED_COMMAND,
        "seeds": SEEDS,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pytest", "torch", "torchgfn")
        },
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "archived_full_grid_evidence": {
            "sha256": sha256(full_grid_path),
            "source_run_id": SOURCE_FULL_GRID_RUN,
            "source_git_sha": SOURCE_FULL_GRID_COMMIT,
            "source_elapsed_seconds": full_grid["elapsed_seconds"],
            "replayed_not_retrained": True,
        },
        "runtime_seconds": {
            "exact": exact_seconds,
            "neural": neural_seconds,
            "independent_checker": checker_seconds,
            "molecule_prerequisite_audit": molecule_seconds,
            "claim_evidence_checkers": claim_checker_seconds,
            "report_figures": report_figure_seconds,
        },
    }
    write_json(ARTIFACTS / "run_metadata.json", metadata)

    source_audit = (ROOT / "docs" / "SOURCE_AUDIT.md").read_text()
    for claim_id, claim in CLAIMS.items():
        claim_dir = ARTIFACTS / f"claim-{claim_id}"
        claim_dir.mkdir(parents=True, exist_ok=True)
        verdict, assessment = verdicts[claim_id]
        write_json(
            claim_dir / "claim_contract.json",
            {
                "claim": claim_id,
                "paper_result": claim["paper_result"],
                "source_anchor": claim["anchor"],
                "machine_checkable_contract": claim["contract"],
                "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
            },
        )
        (claim_dir / "source_audit.md").write_text(source_audit)
        if claim_id in (2, 5, 6):
            method_scope = (
                "The immutable output of source run "
                f"{SOURCE_FULL_GRID_RUN} is replayed and independently checked. "
                "That parent run trained the paper's published 32x32 architecture "
                "for 20,000 steps over three deterministic seeds and enumerated "
                "terminal distributions exactly over all 1,024 states. The "
                f"archived raw JSON has SHA-256 {ARCHIVED_FULL_GRID_SHA256}."
            )
        elif claim_id in (3, 4):
            method_scope = (
                "A fail-closed prerequisite audit checks the released source, "
                "data, checkpoints, locked imports, exact v1 thresholds, and "
                "required comparator surfaces. Missing prerequisites produce "
                "BLOCKED, never a proxy PASS."
            )
        else:
            method_scope = "The accepted exact full-grid theorem regression is rerun."
        (claim_dir / "method.md").write_text(
            f"# Method\n\n{method_scope}\n\n"
            f"Contract: {claim['contract']}\n\n"
            "The fixed campaign reruns accepted regressions and independent "
            "checkers. Expensive raw training is provenance-linked and "
            "content-addressed rather than repeated in this cumulative node. "
            "Child branches vary committed code/config, never the command or "
            "locked environment.\n"
        )
        (claim_dir / "limitations_and_deviations.md").write_text(
            f"# Limitations and deviations\n\n{assessment}\n"
        )
        (claim_dir / "command.txt").write_text(FIXED_COMMAND + "\n")
        write_json(claim_dir / "seeds.json", SEEDS)
        write_json(claim_dir / "run_metadata.json", metadata)
        (claim_dir / "independent_checker.txt").write_text(checker_log)
        write_json(
            claim_dir / "negative_control.json",
            exact["negative_controls"],
        )
        (claim_dir / "EVAL.md").write_text(
            f"# Claim {claim_id}: {verdict}\n\n"
            f"Paper: {claim['paper_result']}\n\n"
            f"Observed baseline: {assessment}\n\n"
            f"Source: {claim['anchor']}.\n"
        )
        if claim_id in (2, 5, 6):
            (claim_dir / "independent_checker.json").write_text(
                claim_checker_logs[claim_id]
            )
            (claim_dir / "negative_control.json").write_text(
                claim_negative_logs[claim_id]
            )

    shutil.copy2(neural_path, ARTIFACTS / "claim-2" / "raw_neural_summary.json")
    shutil.copy2(neural_path, ARTIFACTS / "claim-5" / "raw_neural_summary.json")
    shutil.copy2(
        full_grid_path,
        ARTIFACTS / "claim-5" / "full_grid_raw.json",
    )
    shutil.copy2(
        full_grid_path,
        ARTIFACTS / "claim-6" / "full_grid_raw.json",
    )
    (ARTIFACTS / "claim-1" / "runner_output.txt").write_text(exact_log)
    (ARTIFACTS / "claim-2" / "runner_output.txt").write_text(neural_log)
    (ARTIFACTS / "claim-2" / "full_grid_provenance.txt").write_text(
        f"source_run_id={SOURCE_FULL_GRID_RUN}\n"
        f"source_git_sha={SOURCE_FULL_GRID_COMMIT}\n"
        f"raw_sha256={ARCHIVED_FULL_GRID_SHA256}\n"
        f"source_elapsed_seconds={full_grid['elapsed_seconds']}\n"
        "cumulative_action=replay_and_independent_check\n"
    )
    (ARTIFACTS / "claim-3" / "molecule_audit_runner_output.txt").write_text(
        molecule_log
    )
    (ARTIFACTS / "claim-4" / "molecule_audit_runner_output.txt").write_text(
        molecule_log
    )

    campaign_summary = {
        "paper": "2602.21565v1",
        "baseline_score": "4/12",
        "claims": [
            {
                "claim": claim_id,
                "verdict": verdicts[claim_id][0],
                "assessment": verdicts[claim_id][1],
            }
            for claim_id in sorted(CLAIMS)
        ],
        "claim_1_metrics": {
            "settings": len(exact["claim_2"]["settings"]),
            "max_l1": exact["claim_2"]["max_l1"],
            "max_flow_residual": exact["claim_2"]["max_flow_certificate_residual"],
        },
        "elapsed_seconds": time.perf_counter() - campaign_start,
        "full_grid_raw_sha256": sha256(full_grid_path),
        "full_grid_source_run_id": SOURCE_FULL_GRID_RUN,
        "full_grid_summary": full_grid["summary"],
    }
    write_json(ARTIFACTS / "campaign_summary.json", campaign_summary)

    _, verifier_seconds = run_checked(
        [
            sys.executable,
            "repro/src/verify_campaign.py",
            "--artifacts",
            str(ARTIFACTS),
        ],
        "fail-closed campaign verifier",
    )
    campaign_summary["verifier_seconds"] = verifier_seconds
    write_json(ARTIFACTS / "campaign_summary.json", campaign_summary)

    print("\n===== CAMPAIGN_EVIDENCE_JSON =====")
    print(json.dumps(campaign_summary, indent=2))


if __name__ == "__main__":
    main()
