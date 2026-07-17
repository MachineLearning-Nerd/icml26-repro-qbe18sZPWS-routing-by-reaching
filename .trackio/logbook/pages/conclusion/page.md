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
