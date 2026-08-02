# Paper, verdict, and scope audit

Retrieved on 2026-07-23 with the explicit user agent
`OpenResearch-Reproduction/1.0 (paper audit; contact via GitHub MachineLearning-Nerd)`.

| Source | Immutable locator | SHA-256 |
|---|---|---|
| Judged paper | `https://arxiv.org/pdf/2602.21565v1` | `3a8c57d9f739df274b21a65461db0cce87197d989721c67d118398649b5c9ac3` |
| Current paper | `https://arxiv.org/pdf/2602.21565v3` | `c126395ba290db35973a5bd3952daf1016696c203aaa4f67de805b99cb2abf41` |
| Current HTML | `https://ar5iv.labs.arxiv.org/html/2602.21565` | `52684605ba5a9992ca16ee1f9ecf56d75c92e2b152d5242f48a3424e3d012543` |
| Live verdict dataset | `ICML-2026-agent-repro/verdicts@main`, filtered only by `space_id == "DineshAI/qbe18sZPWS"` | `ff8bcc5eee67ae172d74d292df6f87691ef324380db0f3cc467a3e965efc77bc` |

The judged claims match arXiv v1. The current v3 changed several empirical
numbers and added uncertainty estimates. For example, v1 Table 3 reports
QM9 GAP-SA reward `0.876` (ours), `0.816` (MOGFN), and `0.805` (HN-GFN);
v3 reports `0.873±0.003`, `0.799±0.022`, and `0.742±0.035`. This campaign
tests the exact v1 claims used by the judge and records v3 only as version
context.

## Exact anchors and quantifiers

- Claim 1: v1 Proposition 4.1, Section 4.2, and Appendix A.1. Given `k`
  GFlowNets on a shared state graph with terminal distributions
  `p_i(x) ∝ R_i(x)` and non-negative weights `ω_i`, Eq. (4) at `β=1`
  realizes `p_M(x) ∝ Σ_i ω_i R_i(x)`. Weights need not sum to one.
- Claims 2 and 5: v1 Section 5.1–5.2 and Table 1. The domain is the full
  32×32 HyperGrid. For each `k∈{2,3,4,5}`, results average L1 over 128
  evenly spaced simplex preferences. Ours is `0.003`; MOGFN is
  `0.021/0.027/0.042/0.048`; HN-GFN is
  `0.017/0.021/0.032/0.035`; the no-reaching ensemble is
  `0.117/0.098/0.113/0.111`.
- Claim 3: v1 Section 6.1–6.2 and Table 3. Atom-based QM9 GAP-SA uses
  `β=32`, 10 evenly spaced two-objective preferences, 128 generated
  candidates per preference, and mean scalarized reward of each
  preference's top 10. The reported aggregate is `0.876` for ours,
  `0.816` for MOGFN, and `0.805` for HN-GFN.
- Claim 4: v1 Section 6.2–6.3, Tables 4–5. Logical operators use 5,000
  samples for target-bin accuracy. Timing averages 1,000 samples.
  Fragment logical composition takes `24.70 ms` (2 objectives) and
  `27.02 ms` (3) for ours versus `1789.11 ms` and `1974.64 ms` for
  classifier guidance, with Table 4 reporting higher target-bin percentages
  for ours in every listed fragment and QM9 composition.
- Claim 6: v1 Section 5.3 and Figure 5 (extended in appendix panels). The
  distortion is `δ(x)=u_M(x)/N_M(x)`. The paper says it is not exactly
  constant for nonlinear operators but stays close to `1/Z_M` where the
  composition value `G(·)` is large; deviations should concentrate where
  `G(·)` is small. Figure 5 defines outliers using
  `[Q1−1.5·IQR, Q3+1.5·IQR]`.

## Existing accepted evidence

The existing exact tabular suite is retained as a theorem-level regression.
It enumerates all 1,024 terminal states and triangulates forward rollout with
an independent edge-flow certificate. It does not substitute for the paper's
trained-neural or molecule experiments.

## Environment reconciliation

The official README describes two independent Python 3.10 environments.
Its molecule dependency `gflownet@v0.2.0` pins `torch-geometric==2.4.0`,
whereas the grid dependency `torchgfn==2.4.1` requires
`torch-geometric>=2.6.1`. The campaign requirement is exactly one repository
`.venv`, so the lock uses `torch-geometric==2.6.1` for both codebases. This is
an explicit compatibility override, not an experimental variable; import and
behavior checks must pass before molecule evidence is accepted.

The same README says `torch==2.1.2`, but the published
`torchgfn==2.4.1` package metadata requires `torch>=2.6.0`. The single lock
therefore pins `torch==2.6.0` and treats the paper's upstream GFlowNet commit
as the fixed code source. This dependency-only deviation is tested against
the full accepted grid regression and explicit molecule import/checkpoint
smoke tests.
