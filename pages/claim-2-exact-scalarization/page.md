# Claim 2 — Full neural HyperGrid (legacy route mirror)

This retained route mirrors the current official Claim 2 page. It is not the
older exact-tabular Proposition 4.1 audit, which belongs only to Claim 1.

## Result: exact `0.003` point falsified; qualitative advantage verified

The locked source run trained the published neural architecture and both neural
comparators for 20,000 steps on the paper's 32×32 grid. It enumerated all 1,024
terminal states at 128 preferences for each objective count and repeated the
experiment at seeds `604`, `1337`, and `20260719`.

| Objectives | Paper ours | Reproduced ours (95% t CI) | MOGFN reproduced | HN-GFN reproduced |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.003 | 0.01082 (0.00508–0.01656) | 0.02712 | 0.02788 |
| 3 | 0.003 | 0.00793 (0.00419–0.01167) | 0.03151 | 0.02400 |
| 4 | 0.003 | 0.00724 (0.00340–0.01109) | 0.04292 | 0.03906 |
| 5 | 0.003 | 0.00711 (0.00553–0.00869) | 0.05556 | 0.04119 |

The proposed method beats both independently trained comparators in all 12
seed/objective cells. The exact printed `0.003` point lies below every 95%
interval. If it is instead treated as an unspecified three-decimal rounding,
the `k=4` lower interval overlaps `[0.0025, 0.0035)`, so the stronger rounding
interpretation is not claimed.

The immutable 48-model raw archive is
[`full_grid_raw.json`](../../evidence/full_grid_raw.json), SHA-256
`d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`.
The independent checker is
[`independent_checker_v3.json`](../../evidence/claim-2/independent_checker_v3.json)
and the raw-inconsistent mutation is
[`negative_control_v3.json`](../../evidence/claim-2/negative_control_v3.json).
The complete current page is [Claim 2 — Full neural hypergrid](#/claim-2-full-neural-hypergrid).
