# Claim 5 — Reaching-probability ablation

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c5_contract", "created_at": "2026-08-02T11:49:48+00:00", "title": "Literal claim and verdict contract"}
-->
**Claim.** Removing the reaching-probability weighting term yields L1 `0.098–0.117`, substantially worse than the full policy's `0.003`, showing that reaching probability is essential ([Section 5, Table 1](https://arxiv.org/pdf/2602.21565v1)). The paired test removes only `uᵢ(s)` from the same neural ingredients used for Claim 2, evaluates all 512 preference settings over the same three seeds, requires improvement of at least `0.05` in every seed/objective cell, and requires the ablation aggregate to be within `0.03` of the paper.

**Result: verified.** The no-reaching ensemble is worse in every one of the 12 paired seed/objective cells. Its means are `0.1030–0.1239`, close to the reported range, while the full-policy means are `0.0071–0.0108`. Each paired difference exceeds `0.073`.

| Objectives | Paper no-reaching | Reproduced no-reaching | Reproduced full policy |
| ---: | ---: | ---: | ---: |
| 2 | 0.117 | 0.12389 | 0.01082 |
| 3 | 0.098 | 0.10490 | 0.00793 |
| 4 | 0.113 | 0.11806 | 0.00724 |
| 5 | 0.111 | 0.10304 | 0.00711 |

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_c5_checker", "created_at": "2026-08-02T11:49:48+00:00", "title": "Paired ablation checker and mutation control", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=claim_5_independent_checker.txt
same immutable 48-model archive: PASS
same 32x32 scale and 20,000-step training: PASS
512 preferences over three fixed seeds: PASS
no-reaching aggregate within 0.03 of paper in k=2..5: PASS
paired improvement >= 0.05 in all 12 cells: PASS
verdict: VERIFIED

mutation control: replace no-reaching observations with full-policy values
paired improvement >= 0.05: FAIL
raw aggregate identity: FAIL
mutation rejected: PASS
````

The independent record is [`independent_checker.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-5/independent_checker.json); the condition-destroying mutation is [`negative_control.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-5/negative_control.json). Both bind to raw SHA-256 `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c5_reproduce", "created_at": "2026-08-02T11:49:48+00:00", "title": "Reproduce and interpretation"}
-->
Run `uv sync --frozen && uv run python repro/src/run_campaign.py`. This verifies the ablation comparison even though Claim 2's exact `0.003` value did not reproduce: the causal contrast here is paired against the observed full policy, and the no-reaching degradation is large in every cell. Code and raw outputs are in the [public reproduction repository](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching), pinned to the [author release](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28).
