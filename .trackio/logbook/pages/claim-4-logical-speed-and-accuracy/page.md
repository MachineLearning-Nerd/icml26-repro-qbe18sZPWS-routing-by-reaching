# Claim 4 — Logical speed and accuracy

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c4_contract", "created_at": "2026-08-02T11:49:48+00:00", "title": "Literal claim and verdict contract"}
-->
**Claim.** For harmonic-mean and contrast composition, the method takes `24–27 ms` per sample versus classifier guidance's `1789–1974 ms`—a `40–70×` speed advantage—while achieving comparable or better target-bin accuracy ([Sections 6.2–6.3, Tables 4–5](https://arxiv.org/pdf/2602.21565v1)). The locked test requires both methods on the official molecule sampler, 1,000 timed samples after warmup, and target-bin accuracy from 5,000 samples for each operator.

**Result: BLOCKED — essential material unavailable.** The released source includes the proposed logical sampler but no molecule classifier-guidance implementation, checkpoint, or inference-timing benchmark. It also uses QM9 SA/QED bin thresholds `0.3/0.3`, while arXiv v1 specifies `0.4/0.4`. Without the comparator and exact threshold protocol, neither the speed ratio nor accuracy comparison can be reproduced faithfully.

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_c4_audit", "created_at": "2026-08-02T11:49:48+00:00", "title": "Comparator and protocol inventory", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=claim_4_prerequisite_audit.txt
proposed logical-composition sampler: PRESENT
molecule classifier-guidance implementation: MISSING
classifier-guidance checkpoint: MISSING
1,000-sample timing benchmark: MISSING
paper v1 QM9 GAP/SA/QED thresholds: [0.85, 0.40, 0.40]
released-code thresholds:               [0.85, 0.30, 0.30]
5,000-sample paired accuracy comparison runnable: NO
verdict: BLOCKED
````

The evidence is source-structural rather than an experimental proxy: [`molecule_prerequisite_audit.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-4/molecule_prerequisite_audit.json) records the exact missing surfaces and threshold line, while [`molecule_audit_checker.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-4/molecule_audit_checker.json) validates the audit. Removing a required blocker causes the mutation control to fail.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c4_scope", "created_at": "2026-08-02T11:49:48+00:00", "title": "Reproducibility boundary"}
-->
Timing only the proposed sampler would not test a relative speed claim, and implementing a new classifier-guidance baseline would not reproduce the paper's hardware, checkpoint, or optimization path. The honest result is therefore blocked rather than a one-sided benchmark. Audit command: `uv sync --frozen && uv run python repro/src/run_campaign.py`. Provenance: [pinned author source](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28) and [public reproduction code](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching).
