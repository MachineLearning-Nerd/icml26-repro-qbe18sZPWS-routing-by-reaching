# Claim 4 — Logical speed and accuracy

## Verdict: BLOCKED

The paper reports logical composition as `40–70×` faster than classifier
guidance while retaining comparable or better target-bin accuracy. The exact
contract requires 1,000 timed post-warmup samples per method and 5,000 samples
per operator for harmonic mean and contrast.

The released molecule tree has no classifier-guidance implementation,
checkpoint, or timing benchmark. It also uses QM9 SA/QED bin thresholds `0.3`,
while arXiv v1 specifies `0.4`. Grid timings or a newly invented classifier
would be a different experiment, so no proxy measurement is reported.

## Fail-closed evidence

- [Claim contract](../../evidence/claim-4/claim_contract.json)
- [Prerequisite and threshold inventory](../../evidence/claim-4/molecule_prerequisite_audit.json)
- [Independent audit checker](../../evidence/claim-4/molecule_audit_checker.json)
- [Rejected blocker mutation](../../evidence/claim-4/molecule_negative_control.json)
- [Evaluation](../../evidence/claim-4/EVAL.md)

