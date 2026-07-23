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
SEEDS = [604, 20260719]


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
            "each k=2..5, report per-seed uncertainty, and directly compare L1."
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
            "and require the full policy to improve in every k group."
        ),
    },
    6: {
        "paper_result": (
            "For nonlinear operators delta(x) stays near 1/Z_M in "
            "high-composition-value regions; deviations occur mainly at low G."
        ),
        "anchor": "arXiv v1, Section 5.3, Figure 3 and Figure A6",
        "contract": (
            "For every paper Figure-3/A6 nonlinear grid composition, directly "
            "compute delta, G, 1/Z_M, IQR outliers, and quantile-binned relative "
            "deviation; high-G bins must be closer to 1/Z_M than low-G bins."
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


def baseline_verdicts(neural: dict) -> dict[int, tuple[str, str]]:
    return {
        1: (
            "VERIFIED",
            "Retained full-scale exact theorem regression: 512/512 settings pass.",
        ),
        2: (
            "BLOCKED",
            "Baseline lacks official MOGFN/HN-GFN training and comparison; "
            f"cached neural ingredients have max target L1 "
            f"{max(v['terminal_L1_to_target'] for v in neural['ingredients'].values()):.3f}.",
        ),
        3: ("BLOCKED", "No trained QM9 ingredient or conditional baseline checkpoints."),
        4: ("BLOCKED", "No molecule classifier-guidance timing or 5,000-sample bin audit."),
        5: (
            "BLOCKED",
            "The exact and custom-neural ablations are controls, not the paper's "
            "official trained-neural Table 1 setup.",
        ),
        6: (
            "BLOCKED",
            "The baseline checks the distortion identity but not direct "
            "high-G constancy metrics on the paper's trained models.",
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
    profile_log, profile_seconds = run_checked(
        [sys.executable, "repro/src/profile_official_grid.py"],
        "official HyperGrid local-CPU profile",
    )

    exact = json.loads(exact_path.read_text())
    neural_path = ROOT / "outputs" / "neural_composition.json"
    neural = json.loads(neural_path.read_text())
    verdicts = baseline_verdicts(neural)

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
        "runtime_seconds": {
            "exact": exact_seconds,
            "neural": neural_seconds,
            "independent_checker": checker_seconds,
            "official_grid_cpu_profile": profile_seconds,
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
        (claim_dir / "method.md").write_text(
            f"# Method\n\nBaseline method for Claim {claim_id}.\n\n"
            f"Contract: {claim['contract']}\n\n"
            "Every accepted check is rerun by the fixed campaign command. "
            "Later child branches may add code/config but not change the command "
            "or locked environment.\n"
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

    shutil.copy2(neural_path, ARTIFACTS / "claim-2" / "raw_neural_summary.json")
    shutil.copy2(neural_path, ARTIFACTS / "claim-5" / "raw_neural_summary.json")
    (ARTIFACTS / "claim-1" / "runner_output.txt").write_text(exact_log)
    (ARTIFACTS / "claim-2" / "runner_output.txt").write_text(neural_log)
    (ARTIFACTS / "claim-2" / "cpu_profile_runner_output.txt").write_text(
        profile_log
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
        "cpu_profile": json.loads(
            (
                ARTIFACTS
                / "claim-2"
                / "cpu-profile"
                / "profile.json"
            ).read_text()
        ),
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
