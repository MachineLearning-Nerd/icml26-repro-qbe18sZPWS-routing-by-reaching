# Claim 1 — Exact linear scalarization

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c1_contract", "created_at": "2026-08-02T11:49:48+00:00", "title": "Literal claim and verdict contract"}
-->
**Claim.** Proposition 4.1 states that, for a linear reward scalarization, the mixing policy exactly recovers `p_M(x) ∝ Σᵢ ωᵢRᵢ(x)` ([Section 4.2 and Proposition 4.1](https://arxiv.org/pdf/2602.21565v1)). Verification required the shared 32×32 DAG at `β=1`, all `k=2,…,5`, 128 fixed preferences per `k`, terminal L1 below `1e-12`, and an independently reconstructed flow-conservation residual below `1e-11`.

**Result: verified.** All 512 composed distributions pass. The maximum terminal L1 is `8.61e-16`; the maximum independent flow residual is `2.27e-13`. This numerically audits every state of the finite construction under the proposition's assumptions; it is not presented as a proof replacement.

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_c1_certificate", "created_at": "2026-08-02T11:49:48+00:00", "title": "Exact full-state certificate and controls", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=claim_1_certificate.txt
settings: 512 / 512
grid: 32 x 32 (1,024 terminal states)
max terminal L1: 8.613986260397333e-16
max flow-conservation residual: 2.2737367544323206e-13
verdict: VERIFIED

condition-relaxing controls:
  omit reaching probability L1: 0.15662801795415238
  omit partition factor L1: 0.009328807638861777
  apply beta=2 policy to beta=1 target L1: 0.07017473043675532
````

The three altered constructions fail by many orders of magnitude, showing that the checker can reject a plausible but incorrect composition. Raw results are in [`raw_exact_summary.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-1/raw_exact_summary.json), the locked contract in [`claim_contract.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-1/claim_contract.json), and controls in [`negative_control.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-1/negative_control.json).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c1_reproduce", "created_at": "2026-08-02T11:49:48+00:00", "title": "Reproduce and provenance"}
-->
Run `uv sync --frozen && uv run python repro/src/run_campaign.py`. The implementation is pinned to the [author source revision](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28); the full reproduction code is in the [public repository](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching). No GPU or external service is used.
