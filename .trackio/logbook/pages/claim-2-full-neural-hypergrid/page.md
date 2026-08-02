# Claim 2 — Full neural hypergrid

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c2_contract", "created_at": "2026-08-02T11:49:48+00:00", "title": "Literal claim and verdict contract"}
-->
**Claim.** Table 1 reports L1 `0.003` for the proposed method on the 32×32 grid for two through five objectives, versus `0.021–0.048` for MOGFN and `0.017–0.035` for HN-GFN ([Sections 5.1–5.2, Table 1](https://arxiv.org/pdf/2602.21565v1)). The locked test trained the published neural architecture for 20,000 steps, trained both comparators, enumerated all 1,024 terminal states for 128 fixed preferences per objective count, and repeated seeds `604`, `1337`, and `20260719`. The exact printed point was falsified only if `0.003` fell below every two-sided 95% t interval; qualitative ordering was checked separately in every seed/objective cell.

**Result: the exact printed point is falsified; the qualitative advantage is verified.** The reproduced method beats both comparators in all 12 paired cells, but its four means are `0.0071–0.0108`, and the exact `0.003` point lies below every 95% interval.

| Objectives | Paper ours | Reproduced ours (95% t CI) | MOGFN reproduced | HN-GFN reproduced |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.003 | 0.01082 (0.00508–0.01656) | 0.02712 | 0.02788 |
| 3 | 0.003 | 0.00793 (0.00419–0.01167) | 0.03151 | 0.02400 |
| 4 | 0.003 | 0.00724 (0.00340–0.01109) | 0.04292 | 0.03906 |
| 5 | 0.003 | 0.00711 (0.00553–0.00869) | 0.05556 | 0.04119 |

The conclusion is not an aggregate-only proxy: every independently trained,
paired seed/objective cell has the paper's architecture and scale. The 12 raw
comparisons are exposed here so the full experiment is visible without
inferring it from the four means:

| Seed | Objectives | Ours L1 | MOGFN L1 | HN-GFN L1 | Ours beats both |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 604 | 2 | 0.011591 | 0.029139 | 0.032578 | yes |
| 604 | 3 | 0.008386 | 0.029846 | 0.025105 | yes |
| 604 | 4 | 0.007536 | 0.038590 | 0.043085 | yes |
| 604 | 5 | 0.006696 | 0.052079 | 0.044401 | yes |
| 1337 | 2 | 0.008222 | 0.022535 | 0.024240 | yes |
| 1337 | 3 | 0.006248 | 0.030461 | 0.019838 | yes |
| 1337 | 4 | 0.005572 | 0.042764 | 0.032618 | yes |
| 1337 | 5 | 0.007844 | 0.055191 | 0.041920 | yes |
| 20260719 | 2 | 0.012645 | 0.029689 | 0.026820 | yes |
| 20260719 | 3 | 0.009155 | 0.034226 | 0.027055 | yes |
| 20260719 | 4 | 0.008623 | 0.047421 | 0.041466 | yes |
| 20260719 | 5 | 0.006791 | 0.059397 | 0.037248 | yes |

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_c2_checker", "created_at": "2026-08-02T11:49:48+00:00", "title": "Independent neural-output checker", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=claim_2_independent_checker.txt
paper scale 32x32: PASS
20,000 training steps: PASS
48 trained models recorded: PASS
128 preferences in each seed/objective cell: PASS
recomputed aggregates match archive: PASS
ours beats MOGFN and HN-GFN in all 12 paired cells: PASS
paper point 0.003 below every 95% interval: PASS
verdict on exact printed point: FALSIFIED
````

The raw archive has SHA-256 `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec` and records the 10 h 36 min source run. The standalone checker recomputes every value above directly from that archive and emits the full paired-cell table. See [`full_grid_raw.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/full_grid_raw.json), [`independent_checker_v3.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-2/independent_checker_v3.json), and the mutation [`negative_control_v3.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-2/negative_control_v3.json), which replaces the observations with the paper point and is rejected because the raw aggregates no longer match.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c2_limits", "created_at": "2026-08-02T11:49:48+00:00", "title": "Scope, limitation, and rerun"}
-->
The verdict is deliberately about the exact printed `0.003` point. If `0.003` is interpreted as an unspecified three-decimal rounding, its interval `[0.0025, 0.0035)` overlaps the reproduced lower confidence limit for `k=4`; the data therefore do not support a stronger all-roundings falsification. Recompute the result with `uv run python repro/src/verify_claim_evidence.py --claim 2 --input evidence/full_grid_raw.json --output /tmp/claim2.json`; add `--negative-control` to demonstrate rejection. Run the deterministic cumulative verifier with `uv sync --frozen && uv run python repro/src/run_campaign.py`. Repeating the full 48-model training is unnecessary for verification because the immutable raw output is content-addressed, but the source-run implementation remains in the [reproduction repository](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching).
