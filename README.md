# Routing by Reaching — full-scale CPU reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/notebooks/routing_by_reaching_reproduction.py)

This repository reproduces six judge-selected claims from *Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation* ([arXiv:2602.21565](https://arxiv.org/abs/2602.21565), v1 evidence contract). The main empirical run trained 48 neural models on the paper’s full 32×32 HyperGrid for 20,000 steps over three seeds using local CPU only.

The strongest result is mixed but clear: Routing by Reaching beats MOGFN and HN-GFN in all 12 seed/objective-count cells, yet its observed L1 is `0.0071–0.0108`, and the paper’s exact `0.003` point lies below every 95% t interval. We therefore mark the exact numerical claim **FALSIFIED while preserving the verified qualitative ordering**. The reaching-weight ablation and primary nonlinear-distortion claim are **VERIFIED**; the two molecule claims are **BLOCKED** by missing released checkpoints and comparator code.

![Observed full-scale HyperGrid results](reports/routing-by-reaching-reproduction/images/headline-table1.png)

Read the [illustrated technical report](reports/routing-by-reaching-reproduction/report.md) or open the [tutorial marimo notebook](notebooks/routing_by_reaching_reproduction.py).

## Claim results

| Claim | Paper | Observed | Assessment |
|---|---|---|---|
| Exact linear composition | Exact target recovery | 512/512 settings; max L1 `8.61e-16` | **VERIFIED** |
| Neural grid Table 1 | Ours `0.003` | Ours `0.0071–0.0108`, still best in 12/12 cells | **FALSIFIED** for exact point |
| QM9 GAP-SA | `0.876` vs `0.816`, `0.805` | Required checkpoints/comparators absent | **BLOCKED** |
| Molecule logical speed/accuracy | `40–70×`, comparable/better accuracy | Classifier-guidance/timing surface absent | **BLOCKED** |
| Reaching ablation | `0.098–0.117` vs `0.003` | `0.1030–0.1239` vs `0.0071–0.0108` | **VERIFIED** |
| Primary nonlinear distortion | Approximately constant in high-G regions | Six of six primary rows pass; 5/36 appendix reversals disclosed | **VERIFIED**, primary scope |

No toy result is presented as full-scale. The approved text-only evidence was
published to the existing [Hugging Face Space revision
`1546ac7f`](https://huggingface.co/spaces/DineshAI/qbe18sZPWS/commit/1546ac7f0d7c98d2077e97db46b3c9b62809ecc4)
and is awaiting a live judge pass. No new score is claimed.

## Experiment log

Every formal node used exactly the same command:

`uv sync --frozen && uv run python repro/src/run_campaign.py`

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public landing page and reader-facing artifacts | Not run as an experiment (publication surface) | Presentation only | — |
| [Frozen 4/12 baseline](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/tree/orx/frozen-4-12-baseline) | Freeze judged state and exact theorem regression | `uv sync --frozen && uv run python repro/src/run_campaign.py` | Claim 1 verified; prior toy/blocked state retained | local CPU, 1m20s |
| [Correct QM9 compatibility audit](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/tree/orx/correct-qm9-hdf-compatibility-audit) | Audit data, checkpoints, imports, source surfaces, and v1 thresholds | `uv sync --frozen && uv run python repro/src/run_campaign.py` | Claims 3–4 blocked with fail-closed evidence | HF CPU, 4m09s |
| [Full neural HyperGrid matrix](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/tree/orx/local-cpu-threaded-full-hypergrid-matrix) | Train 48 paper-scale models and evaluate three seeds | `uv sync --frozen && uv run python repro/src/run_campaign.py` | Full Claim 2/5/6 raw evidence | local CPU, 10h36m |
| [Cumulative evidence gate](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/tree/orx/cumulative-claim-contracts-and-independent-evide) | Independent recomputation, negative controls, all-claim regression | `uv sync --frozen && uv run python repro/src/run_campaign.py` | Six explicit verdicts; campaign verifier passed | local CPU, 2m25s |

HF `cpu-upgrade` was profiled only after local execution proved insufficient for convenient parallelism. It was slower for this workload, so the authoritative training result is the completed local run. No GPU was used.

## Reproduce

The environment is pinned to Python 3.10.18 in `uv.lock`, and the project uses exactly one repository-level `.venv`.

```bash
uv sync --frozen
uv run python repro/src/run_campaign.py
```

The cumulative command reruns the exact theorem checks, unit suite, molecule prerequisite audit, claim-specific independent verifiers, negative controls, and report-figure generation. The 10h36m full neural output is preserved at `repro/evidence/full_grid_raw.json` with SHA-256:

```text
d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec
```

To inspect the notebook locally:

```bash
uv run marimo edit notebooks/routing_by_reaching_reproduction.py
uv run marimo run notebooks/routing_by_reaching_reproduction.py
```

## Evidence map

- `docs/SOURCE_AUDIT.md` — immutable v1 source anchors, assumptions, and quantifiers
- `repro/src/run_campaign.py` — fixed cumulative entrypoint
- `repro/src/verify_claim_evidence.py` — fail-closed Claim 2/5/6 adjudicator
- `repro/src/audit_molecule_claims.py` — Claim 3/4 prerequisite audit
- `repro/evidence/full_grid_raw.json` — content-addressed full neural output
- `reports/routing-by-reaching-reproduction/` — self-contained report and figures
- `notebooks/routing_by_reaching_reproduction.py` — evidence-first tutorial

Official source snapshot: `ml-postech/gflownet-composition`, pinned commit `b82493c8cd9b46a0933ab8f19440aebbf3e14b28`.
