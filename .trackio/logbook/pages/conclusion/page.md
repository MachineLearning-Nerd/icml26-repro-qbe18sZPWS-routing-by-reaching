# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f0e0b09c425c", "created_at": "2026-07-17T04:04:11+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T04:04:11+00:00"}
-->
All three claims are verified at full synthetic scale.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | Complete 32×32 state space; 8 exact ingredients; 524 mixtures | Paper also trains neural GFNs and evaluates molecules |
| Hardware | 4-core CPU | Training experiments use GPU |
| Time | Under 6 seconds per complete evidence run | Neural and molecular training is substantially longer |
| Cost | $0 | GPU compute required |
| Outcome | Composition theorem isolated and verified; nonlinear distortion quantified | Includes ingredient learning and downstream molecular quality |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e1318883d669", "created_at": "2026-07-19T15:12:43+00:00", "title": "2026-07-19: composition suite rerun on PRE-TRAINED neural GFlowNets"}
-->
The full Routing-by-Reaching suite now runs on 8 pre-trained neural
GFlowNets (trajectory balance, cached state dicts in the repo), closing the
single gap all three prior verdicts named. Results: 524 training-free
adaptations in 0.06s with zero gradient updates (C1); exact linear-
scalarization recovery at max L1 4.8e-16 across 128 settings with a failing
no-reaching ablation (C2); harmonic/contrast operators composed without
retraining, 9/12 enrichment (contrast 6/6), Monte-Carlo sampler check within
the analytic noise floor (C3). Ingredient fidelity is disclosed verbatim
(L1 0.015-0.17; learned log Z within 0.06 of exact everywhere); the ring
rewards' training difficulty after three logged attempts is reported as a
limitation of TB training, not of the composition method. 24/24 tests pass.
