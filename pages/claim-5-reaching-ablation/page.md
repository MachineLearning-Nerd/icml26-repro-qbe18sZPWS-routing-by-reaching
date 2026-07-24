# Claim 5 — Reaching-weight ablation

## Verdict: VERIFIED

The paper reports L1 `0.098–0.117` when the reaching-probability weighting term
is omitted, versus `0.003` for the full method.

On the faithful three-seed neural run, the no-reaching ensemble has mean L1:

| k | Full policy | No-reaching ensemble | Paper ensemble |
|---:|---:|---:|---:|
| 2 | 0.01082 | 0.12389 | 0.117 |
| 3 | 0.00793 | 0.10490 | 0.098 |
| 4 | 0.00724 | 0.11806 | 0.113 |
| 5 | 0.00711 | 0.10304 | 0.111 |

The ensemble-minus-full paired improvement is at least `0.05` in every one of
the 12 seed/`k` cells, and every aggregate ensemble result is within absolute
L1 `0.03` of the paper.

## Fail-closed evidence

- [Claim contract](../../evidence/claim-5/claim_contract.json)
- [Independent checker](../../evidence/claim-5/independent_checker.json)
- [Rejected identity mutation](../../evidence/claim-5/negative_control.json)
- [Evaluation](../../evidence/claim-5/EVAL.md)
- [Full raw neural output](../../evidence/full_grid_raw.json)

