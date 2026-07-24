# Claim 2 — Full neural HyperGrid

## Verdict: FALSIFIED for the exact printed `0.003` point

The paper reports L1 `0.003` for Routing by Reaching at `k=2…5`, compared with
`0.021–0.048` for MOGFN and `0.017–0.035` for HN-GFN.

We trained the paper-scale 32×32 models for 20,000 steps over seeds `604`,
`1337`, and `20260719`, then evaluated 128 preferences per `k` by exact dynamic
programming over all 1,024 terminal states.

| k | Ours mean | 95% t interval | MOGFN | HN-GFN | Paper ours |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.01082 | [0.00508, 0.01656] | 0.02712 | 0.02788 | 0.003 |
| 3 | 0.00793 | [0.00419, 0.01167] | 0.03151 | 0.02400 | 0.003 |
| 4 | 0.00724 | [0.00340, 0.01109] | 0.04292 | 0.03906 | 0.003 |
| 5 | 0.00711 | [0.00553, 0.00869] | 0.05556 | 0.04119 | 0.003 |

The exact paper point lies below every interval. The qualitative ordering is
nevertheless verified: ours beats both baselines in all 12 seed/`k` cells.
A three-decimal rounding interval around `0.003` overlaps the lower edge of the
`k=4` interval; the machine contract adjudicates the exact printed point.

## Fail-closed evidence

- [Claim contract](../../evidence/claim-2/claim_contract.json)
- [Independent checker](../../evidence/claim-2/independent_checker.json)
- [Rejected negative mutation](../../evidence/claim-2/negative_control.json)
- [Evaluation](../../evidence/claim-2/EVAL.md)
- [Full raw neural output](../../evidence/full_grid_raw.json)

The raw output SHA-256 is
`d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`.

