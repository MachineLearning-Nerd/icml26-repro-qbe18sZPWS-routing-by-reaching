# Outcome-blind review — candidate v2

Packet manifest SHA-256: `2bb0f125e2cf4b6ddc791092d8d8d9090a8a1efdf71f9e378a02563555239634`

The packet removed author outcome labels and score forecasts while retaining the exact challenge prompt, canonical pages, numeric evidence, pinned ingredient states, metrics-only checker, and poster structure. The sanitized raw derivative records the immutable source SHA separately. Twenty-four cheap deterministic tests pass; Claims 2, 5, and 6 pass all `14/14`, `11/11`, and `15/15` checks, while their mutations are rejected. The canonical validator and strict poster gates pass with eight pages and eight hotspots.

| Claim | Blind score | Evidence-backed defect or basis |
| ---: | ---: | --- |
| 1 | 2/2 | Full-state 32×32 audit covers 512 settings; max L1 `8.61e-16`, flow residual `2.27e-13`; three condition-relaxing controls fail. |
| 2 | 2/2 | Full 48-model paper-scale run beats both comparators in 12/12 cells but does not reproduce the exact printed `0.003`; all 95% intervals are above it. The k=4 rounding caveat is explicit. |
| 3 | 0/2 | Required QM9 ingredient and comparator checkpoints are absent. The prerequisite audit is not experimental evidence for `0.876/0.816/0.805`. |
| 4 | 0/2 | Classifier-guidance code/checkpoint/timing are absent, and released thresholds differ from arXiv v1. A one-sided benchmark cannot test the literal comparison. |
| 5 | 2/2 | Same-ingredient paired ablation covers all 12 cells; every degradation exceeds `0.073`, and the mutation is rejected. |
| 6 | 2/2 | Figure 5/Eq. 9 primary comparisons pass in all six rows without a fitted cutoff; five broader appendix reversals remain disclosed. |

Blind-review total: `8/12` — not a live leaderboard score. This is not a perfect blind review because Claims 3 and 4 remain honestly blocked.

Round 1 failed Claim 6 for an inherited wrong figure number and a post-hoc `0.30` cutoff. Round 2 corrected the anchor to Figure 5, replaced the cutoff with the paper-tied high-/low-`G` and L1-share/mass-share comparisons, regenerated the evidence and poster, and reran the entire review.
