# Claim 6 — Direct high-density distortion (legacy route mirror)

**Claim.** For nonlinear operators, `δ(x)=u_M(x)/N_M(x)` remains approximately
constant in high-density/high-composition-value regions where sampling accuracy
matters most.

**Result: verified at the source-primary scope.** For harmonic mean and
contrast over three neural seeds, the audit records `G`, `u_M`, `N_M`, `δ`, and
induced probability for all 1,024 terminals per row: 6,144 state records.
The high region is fixed before examining `δ` as the 102 largest-`G` states;
the comparator is the 512 smallest-`G` states.

| Seed | Operator | `1/Z_M` | Mean `δ`, high | Mean `δ`, bottom | Normalized variance, high / bottom |
| ---: | --- | ---: | ---: | ---: | ---: |
| 604 | harmonic mean | 3.983598 | 3.741365 | 4.430434 | 0.0594 / 0.7432 |
| 604 | contrast | 1.300653 | 1.539196 | 0.681038 | 0.0465 / 0.3576 |
| 1337 | harmonic mean | 4.051886 | 3.787581 | 5.768836 | 0.1110 / 0.9352 |
| 1337 | contrast | 1.324546 | 1.495886 | 0.437336 | 0.0460 / 0.2399 |
| 20260719 | harmonic mean | 4.113329 | 3.718144 | 7.725027 | 0.0399 / 1.2985 |
| 20260719 | contrast | 1.393136 | 1.421947 | 0.576183 | 0.0058 / 0.2003 |

Every row also has lower high-region median absolute deviation and RMSE, and
its high-region L1-error share is below its target-mass share. The independently
recomputed `δ=u_M/N_M` residual is at most `5.27e-6`; Equation 9's pointwise
residual is at most `2.67e-9`. A deterministic denominator shuffle fails these
identities and the regional tests.

Evidence: [`direct_state_audit.json`](../../evidence/claim-6/direct_state_audit.json),
[`direct_state_samples.csv`](../../evidence/claim-6/direct_state_samples.csv),
[`direct_state_check.json`](../../evidence/claim-6/direct_state_check.json), and
the rejected [`direct_state_negative_control.json`](../../evidence/claim-6/direct_state_negative_control.json).
The 36-row appendix stress audit reverses in five rows, so no broader scope is
claimed. See [Claim 6 — High-density distortion](#/claim-6-high-density-distortion).
