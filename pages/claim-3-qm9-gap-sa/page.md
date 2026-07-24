# Claim 3 — QM9 GAP-SA

## Verdict: BLOCKED

The paper reports mean top-10 GAP-SA reward `0.876` for the proposed method,
`0.816` for MOGFN, and `0.805` for HN-GFN. A faithful test requires the
atom-based QM9 generator at β=32, 10 evenly spaced preferences, 128 candidates
per preference, and the three trained methods.

The released tree contains QM9 data and a GAP property scorer, but it lacks the
QM9 GAP and SA generative ingredient checkpoints and the trained MOGFN/HN-GFN
comparators. The included scorer is not a generative checkpoint. Substituting
a proxy generator would not test the claim.

## Fail-closed evidence

- [Claim contract](../../evidence/claim-3/claim_contract.json)
- [Prerequisite inventory](../../evidence/claim-3/molecule_prerequisite_audit.json)
- [Independent audit checker](../../evidence/claim-3/molecule_audit_checker.json)
- [Rejected blocker mutation](../../evidence/claim-3/molecule_negative_control.json)
- [Evaluation](../../evidence/claim-3/EVAL.md)

`BLOCKED` is not experimental verification and receives no proxy credit.

