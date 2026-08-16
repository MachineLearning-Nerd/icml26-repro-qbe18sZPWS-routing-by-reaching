# Branch audit

## Final reader-facing branch inventory

The current public repository has 15 branches: `main` plus 14 descriptive
lineage branches. The table maps every legacy public ref to its final name and
records the pre-normalization tip. After identity normalization the object IDs
change; the final live tips are checked by [`verify_final.py`](verify_final.py).

| Final branch | Former public ref | Initial tip | Purpose |
| --- | --- | --- | --- |
| `main` | `main` | `c4f57ab82cf30f91c9964f0991a3a833803f1416` | Cumulative reader-facing evidence and publication surface. |
| `audit/molecule-claim-prerequisites` | `orx/audit-molecule-claim-prerequisites` | `cdbec61b72016f8d47c8632ba6721ff4a9790038` | Fail-closed audit of C3/C4 molecule prerequisites. |
| `audit/qm9-hdf-compatibility` | `orx/correct-qm9-hdf-compatibility-audit` | `2c7edfaba2b033d3ae6b83234c3b5fec8e2a6b42` | QM9 data and HDF compatibility audit. |
| `evidence/cumulative-claim-contracts` | `orx/cumulative-claim-contracts-and-independent-evide` | `f54e39476739a9184b3712e5308e9e4d7fd74d5e` | Six-claim cumulative gate and independent checkers. |
| `baseline/frozen-4-12` | `orx/frozen-4-12-baseline` | `c927c3af8719e8016cca5b91e47201a0d79ee082` | Immutable historical 4/12 judged baseline. |
| `evidence/full-seeded-hypergrid` | `orx/full-seeded-hypergrid-table-1-and-figure-3` | `1024321fce2d27591c285b9a5c5828b19bf681d7` | Full seeded HyperGrid evidence generation. |
| `experiment/hf-cpu-threaded-hypergrid` | `orx/hf-cpu-threaded-full-hypergrid-matrix` | `6c89eebcb076aab1076272d38a86529d649e2709` | Hugging Face CPU-threaded full-grid run. |
| `experiment/hypergrid-seed-1337` | `orx/hf-seed-1337-faithful-hypergrid-shard` | `65fdb50994188a8cff885bd36e736fd6259e1434` | Faithful seed-1337 HyperGrid shard. |
| `experiment/hypergrid-seed-20260719` | `orx/hf-seed-20260719-faithful-hypergrid-shard` | `1e49b80fd2c9e9ad893d5c08c118833ea500784a` | Faithful seed-20260719 HyperGrid shard. |
| `release/illustrated-report` | `orx/illustrated-report-and-publication-candidate` | `6a25075a85156e2e1838386f01a62295f11f2adb` | Illustrated report and publication candidate. |
| `experiment/local-cpu-threaded-hypergrid` | `orx/local-cpu-threaded-full-hypergrid-matrix` | `6c89eebcb076aab1076272d38a86529d649e2709` | Local CPU-threaded full-grid run; retained separately from HF provenance. |
| `audit/profile-hypergrid-cpu` | `orx/profile-faithful-hypergrid-cpu-path` | `b61fcd6673a353f12fdf6606e1a958767dd8151e` | Profile of the faithful HyperGrid CPU path. |
| `audit/profile-single-thread` | `orx/profile-single-thread-official-hypergrid-worker` | `f2c9252286bb7efe6d1d409527e399daf872510a` | Single-thread official worker profiling. |
| `release/prepublication-report` | `release/prepublication-report` | `968cf06e18d8fd52c629eefc8cbaeb805a79a34f` | Prepublication report and release boundary. |
| `release/published-space-mirror` | `release/published-space-mirror` | `30f73b102aea4f1b6208c8b844cf270fbc8861c9` | Mirror of the additive published Space evidence. |

## Naming and identity policy

- **Final repository:** `MachineLearning-Nerd/icml26-routing-by-reaching-audit`.
- **Default branch:** `main`.
- **Legacy refs removed:** all `orx/*` refs and the old release names listed
  above.
- **Reachable attribution:**
  `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>` for
  both author and committer.
- **Co-author trailers:** none.

The old names remain in this file only as provenance. They are not final
published refs.
