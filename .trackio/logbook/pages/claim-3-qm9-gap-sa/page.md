# Claim 3 — QM9 GAP-SA

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c3_contract", "created_at": "2026-08-02T11:49:48+00:00", "title": "Literal claim and verdict contract"}
-->
**Claim.** On atom-based QM9 GAP-SA, the method attains mean reward `0.876`, exceeding MOGFN `0.816` and HN-GFN `0.805` ([Sections 6.1–6.2, Table 3](https://arxiv.org/pdf/2602.21565v1)). A faithful test requires `β=32`, ten evenly spaced GAP-SA preferences, 128 candidates per preference, each preference's top-10 scalarized reward, and the same aggregate for all three trained methods.

**Result: BLOCKED — essential material unavailable.** The released tree contains `qm9.h5`, an MXMNet GAP property scorer, ingredient-training code, and the proposed sampler. It does not contain the QM9 GAP and SA ingredient GFlowNet checkpoints or the trained MOGFN and HN-GFN GAP-SA comparators. The scorer checkpoint is not a generative GFlowNet checkpoint. Retraining these missing systems would be a new GPU experiment, not a CPU-feasible reproduction within the campaign limit.

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_c3_audit", "created_at": "2026-08-02T11:49:48+00:00", "title": "Fail-closed molecule prerequisite audit", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=claim_3_prerequisite_audit.txt
QM9 data present: PASS
MXMNet GAP scorer present: PASS (property scorer, not a GFlowNet)
QM9 GAP ingredient checkpoint: MISSING
QM9 SA ingredient checkpoint: MISSING
trained QM9 MOGFN GAP-SA checkpoint: MISSING
trained QM9 HN-GFN GAP-SA checkpoint: MISSING
exact 10-preference x 128-candidate comparison runnable: NO
verdict: BLOCKED
````

The complete inventory is [`molecule_prerequisite_audit.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-3/molecule_prerequisite_audit.json). Its independent checker verifies all claim quantifiers and blockers; a mutation that removes the checkpoint blocker is rejected in [`molecule_negative_control.json`](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/claim-3/molecule_negative_control.json).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_c3_scope", "created_at": "2026-08-02T11:49:48+00:00", "title": "Why no proxy result is reported"}
-->
Running the property scorer on arbitrary molecules, substituting grid models, or comparing untrained baselines would not measure the claim. The page therefore preserves the literal `0.876/0.816/0.805` contract and reports no proxy score. Source provenance is the [pinned author release](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28); the audit is rerunnable with `uv sync --frozen && uv run python repro/src/run_campaign.py`.
