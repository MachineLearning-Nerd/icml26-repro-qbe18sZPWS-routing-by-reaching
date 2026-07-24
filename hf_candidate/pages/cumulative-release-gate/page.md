# Cumulative release gate

## Candidate state — not yet published

The cumulative local CPU run used the fixed command:

```text
uv sync --frozen && uv run python repro/src/run_campaign.py
```

It passed 27 tests, reran the exact Claim 1 regression, validated all
Claim 2/5/6 evidence, rejected all three claim-specific mutations, and
validated the Claim 3/4 blocker audit and its negative mutation.

| Claim | Candidate verdict |
|---:|---|
| 1 | VERIFIED |
| 2 | FALSIFIED for the exact printed point; qualitative ordering verified |
| 3 | BLOCKED |
| 4 | BLOCKED |
| 5 | VERIFIED |
| 6 | VERIFIED on primary Figure 3 scope |

This table does not claim a score increase. Only a future live-judge verdict
can change the recorded 4/12 score.

## Cumulative evidence

- [Campaign summary](../../evidence/campaign_summary.json)
- [Fail-closed verifier output](../../evidence/verifier_output.json)
- [Run metadata](../../evidence/run_metadata.json)
- [Source audit](../../evidence/source_audit.md)

The exact judged revision
`61f1a9cbf8bdb3b3b398f9f552e514c27ae59860` remains the protected base. All
old non-index files remain a byte-identical subset of the candidate tree. The
two navigation files that must change to expose new pages are also retained as
byte-identical protected snapshots, together with the
[judged-tree SHA-256 manifest](../../protected/judged-61f1a9cbf8bdb3b3b398f9f552e514c27ae59860/JUDGED_SPACE_MANIFEST.sha256).
