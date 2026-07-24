# Claim 6 — High-G distortion

## Verdict: VERIFIED on the primary Figure 3 scope

The paper states that nonlinear distortion
`δ(x)=u_M(x)/N_M(x)` remains approximately constant where the composition
value `G` is high and sampling accuracy matters.

The primary Figure 3 contract covers `pCircle1 ⊗ pCircle2` and
`pCircle1 contrast pCircle2`. Across both operators and three seeds, all six
rows have lower median relative deviation in the high-G decile than in the
bottom half, and all high-G deviations are at most `0.30`.

The broader Figure A6 stress audit covers 12 settings × 3 seeds. It agrees in
31/36 rows and has negative Spearman association in 30/36 rows; five rows
reverse the high-versus-low ordering. The verdict is therefore not generalized
beyond the primary source scope.

The `0.30` operational tolerance and primary scope were fixed by the cumulative
checker after source and parent-result inspection, not by a blinded
preregistration. This limitation is explicit.

## Fail-closed evidence

- [Claim contract](../../evidence/claim-6/claim_contract.json)
- [Independent checker and all 36 summaries](../../evidence/claim-6/independent_checker.json)
- [Rejected high/low swap](../../evidence/claim-6/negative_control.json)
- [Evaluation](../../evidence/claim-6/EVAL.md)
- [Full raw neural output](../../evidence/full_grid_raw.json)

