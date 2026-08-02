# Executive summary

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_exec_20260802", "created_at": "2026-08-02T11:49:48+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-08-02T11:49:48+00:00"}
-->
This source-faithful CPU audit resolves four of the six official claims against [arXiv v1](https://arxiv.org/pdf/2602.21565v1) and the author release pinned at [`b82493c`](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28). Claim 1 is verified by an exact full-state audit. A 10 h 36 min, 48-model neural reproduction falsifies Claim 2's exact printed `0.003` point while preserving its qualitative advantage over both trained baselines. The same paired run verifies Claim 5. For Claim 6, a fresh 34-minute rerun reproduced six ingredient checkpoints byte-for-byte and directly recorded `G(x)`, `u_M(x)`, `N_M(x)`, and `δ(x)` over all 6,144 primary state/operator/seed combinations; every predeclared high-`G` comparison passes. Claims 3 and 4 are blocked because essential molecule/comparator checkpoints and the classifier-guidance timing implementation were not released.

| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | All six official claims; 32×32 grid at paper scale; released molecule prerequisites | Complete QM9 and logical-composition comparisons |
| Hardware | Apple arm64 CPU, 8 logical CPUs; no GPU | GPU training plus unreleased comparator material |
| Compute time | Accepted neural run 10 h 36 min; direct Claim 6 rerun 34 min; deterministic cumulative checks about 90 s | Not estimable from the released material |
| Cost | Local CPU; $0 accepted-evidence cloud charge | Not estimated |
| Outcome | C1/C5 verified; C2 exact point falsified; C6 source-primary verified; C3/C4 blocked | Not claimed |

The live judged baseline retained for this additive release is `4/12` at HF revision `8de344e3c4a45398170457c954d2b70dd62dab7c`; the candidate's blind-review score is not described as banked leaderboard credit. The exact feedback repeats content from retained legacy routes, so those paths are now content-consistent with the six ordered official pages. Provenance: [paper](https://arxiv.org/pdf/2602.21565v1), [reproduction repository](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching), [existing Hugging Face Space](https://huggingface.co/spaces/DineshAI/qbe18sZPWS), and [author source pin](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28). No Hugging Face Job, Bucket, model, or dataset repository contributed accepted evidence.

---
<!-- trackio-cell
{"type": "figure", "id": "cell_rbr_poster_20260802", "created_at": "2026-08-02T11:49:48+00:00", "title": "Reproduction poster", "pinned": true, "pinned_at": "2026-08-02T11:49:48+00:00", "poster": true}
-->
````html
<!-- poster_embed.html -->
<iframe src="poster_embed.html" title="Routing by Reaching six-claim reproduction poster" width="100%" height="1120" loading="lazy"></iframe>
````

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_protocol_20260802", "created_at": "2026-08-02T11:49:48+00:00", "title": "Pinned protocol and release lineage"}
-->
The deterministic cumulative command is `uv sync --frozen && uv run python repro/src/run_campaign.py`. It reruns the cheap exact audit, validates both content-addressed neural archives, repeats claim-specific independent checkers and mutation controls, audits molecule prerequisites, regenerates figures, and fails closed on any inconsistency. The immutable full-grid record is [`evidence/full_grid_raw.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/full_grid_raw.json), SHA-256 `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`; the direct Claim 6 state archive is [linked here](https://huggingface.co/spaces/DineshAI/qbe18sZPWS/resolve/main/evidence/claim-6/direct_state_audit.json), SHA-256 `dfe60d7381035df45239aab16d266ae87e2a82625fc2163fac54f43df57c28ef`. Every historical Space path remains present; legacy claim routes now redirect to and mirror the corresponding official claim rather than exposing stale proxy evidence.
