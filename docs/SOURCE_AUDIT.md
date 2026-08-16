# Source and paper audit

## Paper identity and versions

- **ICML/review title:** *Routing by Reaching: Composition of Pre-trained
  GFlowNets for Multi-Objective Generation*.
- **arXiv v1 title:** *Training-free Composition of Pre-trained GFlowNets for
  Multi-Objective Generation*.
- **Authors:** Seokwon Yoon, Youngbin Choi, Seunghyuk Cho, Seungbeom Lee,
  MoonJeong Park, and Dongwoo Kim.
- **Review record:** [OpenReview qbe18sZPWS](https://openreview.net/forum?id=qbe18sZPWS).
- **Paper:** [arXiv:2602.21565](https://arxiv.org/abs/2602.21565).

The judged contracts in this repository are explicitly v1-scoped. The audit
also records that later arXiv versions changed several empirical numbers and
added uncertainty estimates; those later values are not silently substituted
for the v1 contracts.

## Immutable source records

| Source | Locator | Recorded SHA-256 or commit |
| --- | --- | --- |
| Judged paper | `https://arxiv.org/pdf/2602.21565v1` | `3a8c57d9f739df274b21a65461db0cce87197d989721c67d118398649b5c9ac3` |
| Current paper context | `https://arxiv.org/pdf/2602.21565v3` | `c126395ba290db35973a5bd3952daf1016696c203aaa4f67de805b99cb2abf41` |
| Current HTML context | `https://ar5iv.labs.arxiv.org/html/2602.21565` | `52684605ba5a9992ca16ee1f9ecf56d75c92e2b152d5242f48a3424e3d012543` |
| Authors' code | [`ml-postech/gflownet-composition`](https://github.com/ml-postech/gflownet-composition) | commit `b82493c8cd9b46a0933ab8f19440aebbf3e14b28` |
| Full neural raw output | [`evidence/full_grid_raw.json`](../evidence/full_grid_raw.json) | `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec` |
| Direct C6 output | [`evidence/claim-6/direct_state_audit.json`](../evidence/claim-6/direct_state_audit.json) | `dfe60d7381035df45239aab16d266ae87e2a82625fc2163fac54f43df57c28ef` |

## Claim anchors and quantifiers

- **C1:** v1 Proposition 4.1, Section 4.2, and Appendix A.1; at `β=1`,
  reaching-weighted composition targets the weighted sum of ingredient reward
  distributions. The exact regression uses `k=2…5` and 128 preferences per
  objective count on the full `32×32` HyperGrid.
- **C2 and C5:** v1 Sections 5.1–5.2 and Table 1; the neural campaign trains
  the published architecture for 20,000 steps over seeds `604`, `1337`, and
  `20260719`, evaluates 128 preferences for each `k`, and retains all 48 model
  records. C2 adjudicates the exact printed `0.003`; C5 removes only the
  reaching term.
- **C3:** v1 Sections 6.1–6.2 and Table 3; the contract requires atom-based
  QM9 GAP-SA, 10 preferences, 128 candidates per preference, and ours/MOGFN/
  HN-GFN checkpoints. Missing required material yields `BLOCKED`.
- **C4:** v1 Sections 6.2–6.3 and Tables 4–5; the contract requires 1,000
  timing samples, 5,000 target-bin samples, classifier guidance, and the v1
  thresholds. Missing comparator/timing material yields `BLOCKED`.
- **C6:** v1 Section 5.3 and Figure 5; the direct audit computes
  `δ(x)=u_M(x)/N_M(x)` and compares high-`G` and low-`G` regions for the two
  primary nonlinear operators across three seeds. The 12-operator appendix
  stress test is reported separately and includes five reversals.

## Environment reconciliation

The authors' README describes separate molecule and grid environments. The
reproduction lock intentionally uses one Python 3.10 environment with
`torch==2.6.0`, `torchgfn==2.4.1`, and `torch-geometric==2.6.1`; these are
explicit compatibility overrides recorded in `pyproject.toml` and `uv.lock`,
not hidden experimental variables. Molecule evidence is accepted only after
the prerequisite/import checks pass, and in the current contract the required
checkpoints/comparators remain absent.

## Evidence boundary

The exact tabular regression, archived neural run, direct state audit, and
negative controls support the declared finite contracts. They do not prove
universal GFlowNet composition theorems, reproduce unavailable molecule
checkpoints, or convert the five broader C6 reversals into a positive universal
claim.
