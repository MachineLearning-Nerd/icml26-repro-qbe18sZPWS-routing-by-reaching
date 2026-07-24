# Pre-publication release report

**Prepared:** 2026-07-24 · **Target Space:** `DineshAI/qbe18sZPWS` · **Status:** awaiting explicit publication approval

## Immutable starting state

| Field | Value |
|---|---|
| Baseline judged score | `4/12` |
| Judged HF Head | `61f1a9cbf8bdb3b3b398f9f552e514c27ae59860` |
| Judge Head | `61f1a9cbf8bdb3b3b398f9f552e514c27ae59860` |
| Baseline Git SHA | `75401ce487eb0c5a759eb583973c57cb455b7b61` |
| Paper evidence contract | `arXiv:2602.21565v1` |
| Paper v1 PDF SHA-256 | `3a8c57d9f739df274b21a65461db0cce87197d989721c67d118398649b5c9ac3` |

No score increase is claimed. Only a new live-judge verdict can change `4/12`.

## Winning evidence and publication surface

- **Scientific evidence branch:** `orx/cumulative-claim-contracts-and-independent-evide`
- **Scientific evidence SHA:** `f54e39476739a9184b3712e5308e9e4d7fd74d5e`
- **Cumulative run:** `54806ee5-bb7c-4255-84cc-36b090052424` — local CPU, 2m25s, passed
- **Publication-candidate branch:** `orx/illustrated-report-and-publication-candidate`
- **Publication-candidate SHA/run:** `6a25075a85156e2e1838386f01a62295f11f2adb` / `19d4a988-ec93-4552-9919-55d98b381d71` — local CPU, 1m25s, passed
- **GitHub main:** confirmed by `git ls-remote` at `6a25075a85156e2e1838386f01a62295f11f2adb`

The fixed run command was identical at every formal node:

```text
uv sync --frozen && uv run python repro/src/run_campaign.py
```

## Experiment tree

```text
Frozen 4/12 baseline (c927c3a; local, passed)
├── Profile faithful HyperGrid CPU path (b61fcd6)
│   └── Full seeded HyperGrid Table 1/Figure 3 (1024321; obsolete HF record stale)
│       └── Local CPU threaded full matrix (6c89eeb; 10h36m, passed)
│           ├── HF CPU full matrix (cancelled after slower profile)
│           ├── HF seed shards (cancelled)
│           └── Cumulative claim evidence gate (f54e394; passed)
│               └── Illustrated publication candidate (6a25075; passed)
└── Molecule prerequisite audit
    └── Correct QM9 HDF compatibility audit (2c7edfa; local/HF checks passed)
```

The authoritative full-scale neural output came from local run
`fd92e2bc-ea94-4004-a820-62abf3e5e917`, commit
`6c89eebcb076aab1076272d38a86529d649e2709`, after 10h36m. Its raw JSON SHA-256
is `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`.

## Claim-by-claim result

| Claim | Candidate verdict | Direct evidence |
|---:|---|---|
| 1 | **VERIFIED** | 512/512 exact settings; max L1 `8.61e-16`; max flow residual `2.27e-13` |
| 2 | **FALSIFIED** for exact printed `0.003` | observed ours `0.0071–0.0108`; `0.003` below every 95% t CI; ours still beats both baselines in all 12 seed/`k` cells |
| 3 | **BLOCKED** | nine required checkpoints absent; molecule MOGFN/HN-GFN comparator surfaces absent |
| 4 | **BLOCKED** | classifier-guidance/timing surfaces absent; released thresholds `0.3` differ from v1 `0.4` |
| 5 | **VERIFIED** | no-reaching ensemble worse by at least `0.05` in all 12 paired cells; aggregate `0.1030–0.1239` |
| 6 | **VERIFIED on primary Figure 3 scope** | six of six primary rows have lower high-G deviation and high-G deviation ≤ `0.30`; broader stress audit passes 31/36 |

Claim 2’s three-decimal rounding interval overlaps the lower edge of the `k=4`
confidence interval. Claim 6’s tolerance/scope is a source-aligned retrospective
contract, not a blinded preregistration. Both limitations are present in the
candidate pages and machine evidence.

## Regression and negative-control gate

- 27 tests passed.
- Claim 1’s omit-reaching, omit-partition, and wrong-β controls were rejected.
- Claim 2’s mutation to exact `0.003` was rejected.
- Claim 5’s mutation making the ensemble identical to the full method was rejected.
- Claim 6’s high/low swap was rejected.
- The molecule audit’s removed-blocker mutation was rejected.
- The final campaign verifier passed every accepted/blocked claim check.
- Five report figures regenerated from raw JSON and passed size checks.
- `marimo check notebooks/routing_by_reaching_reproduction.py` passed.

## Reader-facing artifacts

- `README.md` — public landing page and exact-command experiment log
- `reports/routing-by-reaching-reproduction/report.md` — illustrated technical report
- `notebooks/routing_by_reaching_reproduction.py` — evidence-first tutorial notebook
- `hf_candidate/` — text-only additive Space overlay
- `release/hf/` — exact upload allowlist, manifests, subset proof, and validation

The report is also mirrored into the OpenResearch Files directory.

## Compute runtime and cost

No GPU was used.

### Local CPU

Formal local runs account for approximately **10h52m** wall time, dominated by
the authoritative 10h36m full matrix. External compute charge: **$0**.

### Hugging Face CPU Upgrade

HF CPU Upgrade is [billed per minute at `$0.03/hour`](https://huggingface.co/docs/hub/jobs-pricing). The completed/cancelled
short HF checks account for about 93 recorded minutes (approximately `$0.05`).
One obsolete early full-matrix record is stale at 24h15m; its log and recorded
duration stopped advancing more than eight hours before this report. A fresh
cancellation request was accepted after updating `orx` to 0.1.77, but the local
record did not transition within 60 seconds.

Using the stale record’s entire displayed duration as a conservative upper
bound gives **1,548 billable minutes, at most approximately `$0.78`** across all
HF CPU Upgrade work. The HF invoice is authoritative. The stale job produced no
accepted claim evidence.

## Protected-logbook subset proof

Candidate directory:

```text
/Users/dineshjinjala/Documents/AllCode/ICMLPapers/OpenSearch/files/icml26-repro-qbe18szpws-routing-by-reaching/candidate/space-f54e394-v3
```

Validation:

- all old non-index files are a byte-identical subset of the candidate;
- `logbook.json` and `pages/index.md` change only to expose the new pages;
- byte-identical judged copies of those two navigation files and the full
  judged-tree manifest are included under `protected/judged-61f1…/`;
- all referenced logbook pages exist;
- every changed/new upload is UTF-8 text;
- every JSON file parses;
- no configured secret/token pattern was found;
- the full-grid raw SHA matches;
- candidate validation passed.

Machine-readable proof: `release/hf/SUBSET_CHECK.json` and
`release/hf/VALIDATION.json`.

## Exact Hugging Face upload allowlist

The exact 81-path text-only allowlist is `release/hf/UPLOAD_ALLOWLIST.txt`.
SHA-256 values for exactly those paths are in
`release/hf/UPLOAD_MANIFEST.sha256`; the complete candidate tree is in
`release/hf/CANDIDATE_TREE_MANIFEST.sha256`.

Publication will use an additive Hugging Face API commit to the existing
`DineshAI/qbe18sZPWS` Space. It will abort if the remote head is no longer
`61f1a9cbf8bdb3b3b398f9f552e514c27ae59860`, upload only allowlisted text
files, perform no deletion, and verify the returned revision and every uploaded
SHA afterward.

## Reproduction and state-changing command ledger

### Startup and source audit

```text
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx projects --json
orx project view 6941e3e3-c555-40d5-b488-d20786258500
orx runs 6941e3e3-c555-40d5-b488-d20786258500
git status --short
git rev-parse HEAD
df -h
env | cut -d= -f1 | sort
```

Paper v1/v3 and the live verdict dataset were fetched with explicit
User-Agent/API requests; the exact source URLs, dates, hashes, anchors, and
quantifiers are recorded in `docs/SOURCE_AUDIT.md`. The judged Space revision
was downloaded and manifested before any candidate work.

### Formal experiment launches

```text
orx exp run e6854a1a-1569-463f-a8d5-bb674fc852b5 --backend local
orx exp run 23d6f313-e1de-4c86-9962-2d2d004c4288 --backend local
orx exp run 13176fc6-325b-496d-b1a7-6f1f32a6da77 --backend local
orx exp run 13176fc6-325b-496d-b1a7-6f1f32a6da77 --backend hf --flavor cpu-upgrade
orx exp run 69f08fe8-577f-4941-ba0d-94cdcf93809c --backend hf --flavor cpu-upgrade
orx exp run 2014e9d7-5cb3-423b-a570-23d1abf5a433 --backend local
orx exp run 812f6b89-ba4b-4302-97d9-4a711757a413 --backend hf --flavor cpu-upgrade
orx exp run 8deed544-218b-4ab5-9d2a-2535e71c2a38 --backend local
orx exp run 8deed544-218b-4ab5-9d2a-2535e71c2a38 --backend hf --flavor cpu-upgrade
orx exp run 7de36047-25eb-4090-820a-7a13556f219a --backend local
orx exp run 7de36047-25eb-4090-820a-7a13556f219a --backend hf --flavor cpu-upgrade
orx exp run c1ffc0d2-4d21-4245-b5fb-2fe5c21b33fb --backend hf --flavor cpu-upgrade
orx exp run d711802a-ba20-4419-8122-efc4f5ba8865 --backend hf --flavor cpu-upgrade
orx exp run 17d98114-94a0-4255-84cc-36b090052424 --backend local
orx exp run b2b13187-3f3c-496c-ba5e-a8f43d46ab80 --backend local
```

Every launch was monitored with `orx exp wait`, analyzed with `orx logs`, and
recorded with `orx exp desc`. Cancelled variants used `orx exp cancel`.

### Local validation and publication preparation

```text
uv run pytest -q
uv run marimo check notebooks/routing_by_reaching_reproduction.py
uv run python repro/src/build_report_figures.py --output reports/routing-by-reaching-reproduction/images
uv run python repro/src/prepare_release_candidate.py --protected <judged-snapshot> --overlay hf_candidate --candidate <candidate-v3> --release-dir <release-v3>
shasum -a 256 -c <release-v3>/UPLOAD_MANIFEST.sha256
git push origin HEAD:main
git ls-remote origin refs/heads/main
orx update
orx exp cancel 69f08fe8-577f-4941-ba0d-94cdcf93809c
```

Read-only inspection commands (`orx exp status`, `orx runs`, `orx logs`,
`git diff`, `git status`, `find`, `rg`, `sed`, `jq`, `shasum`, and image
inspection) were used throughout. No unmanaged `pip`, conda, GPU command,
second Space creation, or Hugging Face upload command was executed.

## Approval gate

All claim/evidence, report, manifest, and text-only candidate checks are ready.
The existing Hugging Face Space is still at the judged head. Publication is
stopped here pending one explicit user approval.
