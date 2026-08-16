# ICML 2026 — Routing by Reaching

This repository is a claim-by-claim reproduction audit of [*Routing by
Reaching: Composition of Pre-trained GFlowNets for Multi-Objective
Generation*](https://openreview.net/forum?id=qbe18sZPWS). The arXiv v1 record
uses the title [*Training-free Composition of Pre-trained GFlowNets for
Multi-Objective Generation*](https://arxiv.org/abs/2602.21565). The repository
keeps the paper's positive, negative, and blocked outcomes separate so a
reader can see exactly what the evidence supports.

## Status at a glance

`MIXED_RESULTS_WITH_LIMITATIONS`

The recorded cumulative campaign has six explicit verdicts. Claims 1, 5, and
6 have scoped evidence; Claim 2 is falsified for the exact printed `0.003`
point while retaining the qualitative ordering; Claims 3 and 4 are blocked by
missing material required for a faithful molecule reproduction. No toy or
proxy result is promoted to full-scale evidence.

| Claim | Paper result | Evidence produced here | Assessment |
| --- | --- | --- | --- |
| C1 — Exact linear composition | Exact target recovery | 512/512 full `32×32` HyperGrid settings; maximum L1 `8.61e-16`; independent flow residual `2.27e-13`. | `VERIFIED_SCOPED` |
| C2 — Neural HyperGrid comparison | Ours `0.003` | 48 trained models, 3 seeds, 128 preferences for each `k=2…5`; ours is best in all 12 cells but observed means are `0.0071–0.0108`. | `FALSIFIED_EXACT_POINT` |
| C3 — QM9 GAP-SA | Ours `0.876` vs MOGFN `0.816` and HN-GFN `0.805` | Fail-closed prerequisite audit finds the required ingredient and comparator checkpoints absent. | `BLOCKED` |
| C4 — Logical molecule speed/accuracy | `40–70×` speedup with comparable/better accuracy | Fail-closed prerequisite audit finds no classifier-guidance implementation/checkpoint/timing surface; released thresholds also differ from arXiv v1. | `BLOCKED` |
| C5 — Reaching-weight ablation | No-reaching ensemble `0.098–0.117` vs ours `0.003` | The faithful no-reaching neural ablation is worse in all 12 paired cells; observed ranges are `0.1030–0.1239` vs `0.0071–0.0108`. | `VERIFIED_SCOPED` |
| C6 — Nonlinear distortion | Distortion is closer to constant where `G` is high | Direct `G`, `u_M`, `N_M`, and `δ=u_M/N_M` audit over 6,144 primary state/operator/seed rows; all six primary rows pass. | `VERIFIED_PRIMARY_SCOPE` |

The detailed production paths are in [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md).
The raw records and independent checkers remain under [`evidence/`](evidence/)
and [`repro/evidence/`](repro/evidence/).

## What the paper does

Routing by Reaching composes pre-trained single-objective Generative Flow
Networks at inference time. The mixer weights each ingredient by both the
objective preference and the ingredient's probability of reaching the current
partial state. For linear scalarization this yields an exact target
distribution; for nonlinear operators the paper studies the induced distortion
and applies the method to HyperGrid and molecule-generation tasks.

## Paper record

- **ICML/review title:** *Routing by Reaching: Composition of Pre-trained
  GFlowNets for Multi-Objective Generation*.
- **arXiv v1 title:** *Training-free Composition of Pre-trained GFlowNets for
  Multi-Objective Generation*.
- **Authors:** Seokwon Yoon, Youngbin Choi, Seunghyuk Cho, Seungbeom Lee,
  MoonJeong Park, and Dongwoo Kim.
- **Venue:** ICML 2026.
- **Preprint:** [arXiv:2602.21565](https://arxiv.org/abs/2602.21565).
- **Review record:** [OpenReview qbe18sZPWS](https://openreview.net/forum?id=qbe18sZPWS).
- **Authors' implementation:** [`ml-postech/gflownet-composition`](https://github.com/ml-postech/gflownet-composition),
  pinned by this audit to commit
  [`b82493c`](https://github.com/ml-postech/gflownet-composition/commit/b82493c8cd9b46a0933ab8f19440aebbf3e14b28).

This repository is an independent reproduction and evidence audit. It is not
the authors' implementation. The authors' `grid/` and `mols/` source trees are
retained as the official source snapshot and are tested only where the claim
contract has the required inputs.

## Claim-to-evidence path

```text
paper anchor → source-faithful producer or prerequisite audit
             → content-addressed raw evidence → independent checker/control
             → scoped verdict
```

| Claim | Producer | Evidence and independent check |
| --- | --- | --- |
| C1 | [`repro/src/run_routing_by_reaching.py`](repro/src/run_routing_by_reaching.py) enumerates the exact HyperGrid and evaluates all preferences. | [`evidence/claim-1/`](evidence/claim-1/), [`repro/tests/test_reproduction.py`](repro/tests/test_reproduction.py), flow-conservation and wrong-beta/omission controls. |
| C2 | The archived full neural run trained 48 models for 20,000 steps over seeds `604`, `1337`, and `20260719`; terminal distributions are evaluated over all 1,024 states and 128 preferences per objective count. | [`evidence/full_grid_raw.json`](evidence/full_grid_raw.json), [`evidence/claim-2/independent_checker_v3.json`](evidence/claim-2/independent_checker_v3.json), and the exact-point mutation in [`evidence/claim-2/negative_control_v3.json`](evidence/claim-2/negative_control_v3.json). |
| C3 | [`repro/src/audit_molecule_claims.py`](repro/src/audit_molecule_claims.py) checks whether the released QM9 ingredients, checkpoints, and comparators can satisfy the v1 contract. | [`evidence/claim-3/`](evidence/claim-3/); missing prerequisites produce `BLOCKED`, never proxy credit. |
| C4 | The same fail-closed molecule audit checks classifier guidance, timing, target-bin thresholds, and imports. | [`evidence/claim-4/`](evidence/claim-4); no timing or comparator surface is silently substituted. |
| C5 | The full neural archive is evaluated with only the reaching term removed. | [`evidence/claim-5/`](evidence/claim-5), paired seed/objective cells, and the removed-reaching negative control. |
| C6 | [`repro/src/run_claim_6_direct_audit.py`](repro/src/run_claim_6_direct_audit.py) directly enumerates `G`, `u_M`, `N_M`, `δ`, and region statistics for six primary rows; the stdlib checker independently validates them. | [`evidence/claim-6/direct_state_audit.json`](evidence/claim-6/direct_state_audit.json), [`evidence/claim-6/direct_state_check.json`](evidence/claim-6/direct_state_check.json), and the shuffled-`N_M` control. |

The cumulative gate is [`repro/src/run_campaign.py`](repro/src/run_campaign.py).
Its recorded checker result is [`evidence/verifier_output_v3.json`](evidence/verifier_output_v3.json).

## Reproduce the recorded campaign

The locked environment is Python 3.10.18. The cumulative command is:

```bash
uv sync --frozen
uv run python repro/src/run_campaign.py
```

The campaign replays the exact regression, independent checkers, molecule
prerequisite audit, negative controls, direct Claim 6 checker, and report
figures. The expensive 48-model training output is retained and content-
addressed rather than silently retrained on every cumulative run:

- full neural archive SHA-256:
  `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`;
- direct Claim 6 archive SHA-256:
  `dfe60d7381035df45239aab16d266ae87e2a82625fc2163fac54f43df57c28ef`;
- source run ID: `fd92e2bc-ea94-4004-a820-62abf3e5e917`.

The illustrated report is [`reports/routing-by-reaching-reproduction/report.md`](reports/routing-by-reaching-reproduction/report.md)
and the executable tutorial is [`notebooks/routing_by_reaching_reproduction.py`](notebooks/routing_by_reaching_reproduction.py).

## Historical external evaluation boundary

The repository records external Space snapshots as provenance, not as a new
claim. Different historical files record a `4/12` or `5/12` baseline at
different Hugging Face revisions; the v5 outcome-blind review is `8/12`, not a
live leaderboard score. The candidate release explicitly makes no current
score claim. See [`evidence/release_comparison_v5.json`](evidence/release_comparison_v5.json)
and [`evidence/BLIND_REVIEW_V5.md`](evidence/BLIND_REVIEW_V5.md).

## Repository map

| Path | Role |
| --- | --- |
| [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md) | Claim-by-claim producers, checks, metrics, verdicts, and limitations. |
| [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md) | Mapping from every legacy branch to its final reader-facing name and purpose. |
| [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md) | Paper versions, source hashes, anchors, and dependency deviations. |
| [`STATUS.md`](STATUS.md) | Current collection status and external-evaluation boundary. |
| [`REPORT.md`](REPORT.md) | Short audit decision record. |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | Locked environment and recorded replay provenance. |
| [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md) | Thank-you note to the authors. |
| [`CITATION.cff`](CITATION.cff) | Machine-readable paper and repository citation. |
| [`claims.json`](claims.json) | Machine-readable six-claim ledger. |
| [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json) | Evidence hashes, source, branches, and attribution. |
| [`verify_final.py`](verify_final.py) | Fail-closed final-state verifier. |

## Branches and attribution

The final published branch names are descriptive. `main` is the cumulative
reader-facing surface; the remaining branches preserve experiment, audit,
baseline, and release lineage. See [`BRANCH_AUDIT.md`](BRANCH_AUDIT.md) for the
complete mapping. All reachable commits are normalized to
`MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>` with no
co-author trailers.

## Citation and thanks

Please cite both the paper and this audit when reusing the code or evidence.
See [`CITATION.cff`](CITATION.cff) and [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).
