# STATUS — Routing by Reaching (`qbe18sZPWS`)

**Session:** autoloop. **Last updated:** 2026-07-17. **State:** locally complete;
publication queued.

## Source audit

- Paper: arXiv 2602.21565; OpenReview `qbe18sZPWS`.
- Official code: `ml-postech/gflownet-composition` pinned at
  `b82493c8cd9b46a0933ab8f19440aebbf3e14b28`.
- The official synthetic domain is a finite 32×32 DAG. Exact tabular ingredient
  flows allow full-state verification of the scored composition claims without
  reducing scale or conflating them with neural training error.

## Evidence

- 8 exact ingredients; every policy, `u=F/Z`, and terminal target passes at
  float64 roundoff.
- 512/512 linear settings (128 each for 2, 3, 4, and 5 objectives) match exactly:
  max L1 `8.32e-16`, max pointwise error `4.34e-18`.
- Independent edge-flow conservation certificate passes in every setting.
- 12/12 harmonic/contrast settings are valid and satisfy the paper's distortion
  identity; nonlinear L1 range `0.039–0.288` is reported honestly as approximate.
- 17/17 tests pass; all 3 theorem-assumption controls are rejected.

## Next

- Publish to `DineshAI/qbe18sZPWS` after the daily 20-Space creation quota
  resets, verify public tags/artifact bucket, and request verdict.
