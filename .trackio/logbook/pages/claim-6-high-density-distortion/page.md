# Claim 6 — High-density distortion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c6_contract", "created_at": "2026-08-02T11:49:48+00:00", "title": "Literal claim and verdict contract"}
-->
**Claim.** For nonlinear operators, `δ(x)=u_M(x)/N_M(x)` remains approximately constant in high-density/high-composition-value regions where sampling accuracy matters most ([Section 5.3 and Figure 5](https://arxiv.org/pdf/2602.21565v1)). The source-primary test directly computes `δ`, `G`, `1/Z_M`, target mass, and Eq. 9's L1 contribution for `circle1` harmonic-mean `circle2` and `circle1` contrast `circle2`, over three neural seeds. In each of six rows, the high-`G` decile must have lower median relative deviation than the bottom half and a lower L1-error share than its target-mass share. These are direct comparisons stated by Figure 5 and Eq. 9; no fitted magnitude cutoff is used. All 12 released appendix settings are retained as a broader stress audit, not silently folded into the narrower primary claim.

**Result: verified at the source-primary scope.** All six primary rows pass. High-`G` median relative deviations span `0.022–0.251`; bottom-half deviations span `0.383–0.940`. The high-`G` decile contains about `49–76%` of target mass, while its L1-error share is lower in every row. The broader 36-row stress audit agrees in 31 rows and reverses in five, so this page does not generalize the verdict to every appendix setting.

| Seed | Operator | High-`G` deviation | Bottom-half deviation | High-`G` target mass | High-`G` L1 share |
| ---: | --- | ---: | ---: | ---: | ---: |
| 604 | harmonic mean | 0.186 | 0.460 | 0.758 | 0.591 |
| 604 | contrast | 0.251 | 0.851 | 0.494 | 0.393 |
| 1337 | harmonic mean | 0.186 | 0.383 | 0.759 | 0.710 |
| 1337 | contrast | 0.237 | 0.940 | 0.495 | 0.381 |
| 20260719 | harmonic mean | 0.206 | 0.427 | 0.757 | 0.584 |
| 20260719 | contrast | 0.022 | 0.800 | 0.496 | 0.220 |

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_c6_checker", "created_at": "2026-08-02T11:49:48+00:00", "title": "Source-primary distortion checker", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=claim_6_independent_checker.txt
six exact source-primary Figure 5 rows: PASS
delta/G/normalizer identity and mass sanity: PASS
high-G closer than bottom half in every primary row: PASS
high-G L1-error share below target-mass share in every primary row: PASS
broader appendix audit: 31 / 36 in the same deviation direction
scope limitation preserved: PASS
verdict at source-primary scope: VERIFIED

mutation control: swap high-G and low-G deviations
primary ordering: FAIL
primary error-share comparison: PASS
mutation rejected: PASS
````

The repaired source-scoped result and all 36 broader rows are in [`independent_checker_v2.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-6/independent_checker_v2.json); the bin-swapping control is [`negative_control_v2.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-6/negative_control_v2.json). The [v2 contract](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-6/claim_contract_v2.json) fixes the source anchor and comparison before scoring.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c6_limits", "created_at": "2026-08-02T11:49:48+00:00", "title": "Scope reconciliation and rerun"}
-->
The immutable raw campaign originally applied a stricter universal gate over all appendix settings and correctly marked that exploratory gate blocked. The first source-scoped checker also used a post-hoc `0.30` cutoff; blind review rejected it. The repaired checker resolves the official Figure 5 claim using only direct paper-tied comparisons and leaves the five appendix reversals visible. The records are therefore different scopes and successive repairs, not hidden outcomes. Run `uv sync --frozen && uv run python repro/src/run_campaign.py`. The 48-model raw record is content-addressed in the [public repository](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching).
