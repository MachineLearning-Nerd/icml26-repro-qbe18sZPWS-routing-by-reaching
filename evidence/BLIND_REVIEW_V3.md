# Outcome-blind review — direct-state repair

Packet content-manifest SHA-256: `efc60fceb6997d0ee3f40b7d74e9c977003692007c6e21fae20de266af4818e1`

The final packet contains the exact generated challenge prompt, nine canonical
logbook nodes, numeric evidence, and rerunnable metrics-only checkers. It
excludes campaign notes, score forecasts, desired verdicts, author outcome
labels, historical pages, and all prior reviews. All referenced packet pages
exist. Claim 2's normal checker passes all 14 gates and its paper-point mutation
fails three. Claim 6's direct checker passes all 12 top-level gates and all 60
row-level checks; shuffling `N_M` rejects the state math and three high-`G`
comparisons. The direct archive SHA-256 is
`dfe60d7381035df45239aab16d266ae87e2a82625fc2163fac54f43df57c28ef`.

| Claim | Blind score | Evidence-backed basis or defect |
| ---: | ---: | --- |
| 1 | 2/2 | Full-state 32×32 audit covers 512 settings; maximum L1 `8.61e-16`, flow residual `2.27e-13`; three condition-relaxing controls fail. |
| 2 | 2/2 | Full 48-model paper-scale run exposes all 12 paired seed/objective cells. Ours beats both trained comparators in every cell, but every 95% interval excludes the exact printed `0.003`; the rounding limitation is explicit. |
| 3 | 0/2 | Required QM9 ingredient and MOGFN/HN-GFN comparator checkpoints are absent. The prerequisite audit cannot substitute for the literal `0.876/0.816/0.805` experiment. |
| 4 | 0/2 | Classifier-guidance implementation, checkpoint, and timing path are absent; released thresholds also differ from arXiv v1. A one-sided timing proxy cannot test the claim. |
| 5 | 2/2 | Same-ingredient paired ablation covers all 12 cells; every degradation exceeds `0.073`, and the condition-destroying mutation is rejected. |
| 6 | 2/2 | Six byte-identical checkpoint reruns directly expose `G`, `u_M`, `N_M`, `δ=u_M/N_M`, and induced probability for 6,144 states. Every high-`G` variance, deviation, RMSE, and error-share comparison passes; the shuffled-`N_M` control fails. Five broader appendix reversals remain disclosed. |

Blind-review total: `8/12`. This is not a live leaderboard score. Claims 3 and
4 prevent `PERFECT BLIND REVIEW`; both are evidence-backed material/compute
blockers rather than repairable CPU defects.

The first v3 packet was rejected because it accidentally retained the prior v2
review and therefore leaked an earlier score. The final packet excludes every
prior review, was rebuilt from scratch, rescanned for outcome labels, and was
reviewed and rerun in full.
