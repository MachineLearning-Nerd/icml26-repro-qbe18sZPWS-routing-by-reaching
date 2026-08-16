# Status

- **Collection status:** `MIXED_RESULTS_WITH_LIMITATIONS`.
- **Paper:** *Routing by Reaching: Composition of Pre-trained GFlowNets for
  Multi-Objective Generation*; arXiv v1 `2602.21565` uses the title *Training-
  free Composition of Pre-trained GFlowNets for Multi-Objective Generation*.
- **Authors:** Seokwon Yoon, Youngbin Choi, Seunghyuk Cho, Seungbeom Lee,
  MoonJeong Park, and Dongwoo Kim.
- **Claim vector:** C1 `VERIFIED_SCOPED`; C2
  `FALSIFIED_EXACT_POINT` with qualitative ordering retained; C3 `BLOCKED`;
  C4 `BLOCKED`; C5 `VERIFIED_SCOPED`; C6
  `VERIFIED_PRIMARY_SCOPE`.
- **Recorded cumulative run:** source run
  `fd92e2bc-ea94-4004-a820-62abf3e5e917`, source commit
  `f54e39476739a9184b3712e5308e9e4d7fd74d5e`, locked Python 3.10.18
  environment, content-addressed raw outputs.
- **Independent release gate:** [`evidence/verifier_output_v3.json`](evidence/verifier_output_v3.json)
  reports `passed: true`; claim-specific negative controls are retained.
- **External score boundary:** historical HF and blind-review records are
  preserved, but this GitHub repository makes no current judge-score claim.
- **Author of this audit:** `MachineLearning-Nerd`.

The decisive limitations are the exact numerical mismatch for the printed C2
point, missing molecule checkpoints/comparators for C3, missing classifier-
guidance/timing material for C4, and the five broader appendix reversals kept
outside C6's primary scope.
