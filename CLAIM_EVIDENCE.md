# Claim-to-evidence map

The repository uses three outcomes: `VERIFIED` means the declared finite
contract and controls pass; `FALSIFIED` means the exact contract fails under
its stated falsification rule; `BLOCKED` means required material is absent and
no proxy is accepted.

## Common execution graph

```text
paper anchor → source-faithful producer or prerequisite audit
             → raw/content-addressed artifact → independent checker
             → mutation or negative control → scoped verdict
```

The cumulative entrypoint is [`repro/src/run_campaign.py`](repro/src/run_campaign.py).
The recorded gate is [`evidence/verifier_output_v3.json`](evidence/verifier_output_v3.json).

## C1 — Exact linear composition

- **Anchor:** v1 Proposition 4.1, Section 4.2, Appendix A.1.
- **Producer:** [`repro/src/run_routing_by_reaching.py`](repro/src/run_routing_by_reaching.py)
  constructs the exact HyperGrid and evaluates the reaching-weighted mixture.
- **Contract:** `32×32` grid; `k=2…5`; 128 simplex preferences per `k`; exact
  terminal L1 and independent flow-conservation checks.
- **Evidence:** [`evidence/claim-1/`](evidence/claim-1/),
  [`repro/tests/test_reproduction.py`](repro/tests/test_reproduction.py), and
  the cumulative exact summary.
- **Result:** 512/512 settings pass; maximum L1 `8.613986260000000e-16` and
  maximum flow residual `2.2737367544323206e-13`.
- **Controls:** omit reaching, omit partition normalization, and use the wrong
  beta; each mutation is rejected by the cumulative gate.
- **Assessment:** `VERIFIED_SCOPED`; this is a finite exact regression, not a
  proof of the universal theorem.

## C2 — Neural HyperGrid comparison

- **Anchor:** v1 Sections 5.1–5.2 and Table 1.
- **Producer:** the archived parent run trains the published `32×32` neural
  architecture for 20,000 steps over seeds `604`, `1337`, and `20260719`.
  It retains eight ingredient GFNs, MOGFN, and HN-GFN records: 48 models total.
  Every terminal distribution is evaluated over 1,024 states and 128
  preferences for each `k=2…5`.
- **Evidence:** [`evidence/full_grid_raw.json`](evidence/full_grid_raw.json),
  [`evidence/claim-2/independent_checker_v3.json`](evidence/claim-2/independent_checker_v3.json),
  and [`evidence/claim-2/negative_control_v3.json`](evidence/claim-2/negative_control_v3.json).
- **Falsification rule:** the exact printed `0.003` must lie below every
  two-sided 95% t interval; the independent checker also requires ours to beat
  both baselines in every seed/objective cell.
- **Result:** ours is best in all 12 cells, but its observed means are about
  `0.0108`, `0.0079`, `0.0072`, and `0.0071` for `k=2…5`; the exact `0.003`
  point is below every interval. The qualitative ordering is retained.
- **Assessment:** `FALSIFIED_EXACT_POINT`; this does not falsify the method's
  qualitative advantage or every rounded interpretation of the table.

## C3 — QM9 GAP-SA

- **Anchor:** v1 Sections 6.1–6.2 and Table 3.
- **Producer:** [`repro/src/audit_molecule_claims.py`](repro/src/audit_molecule_claims.py)
  checks the released source, QM9 data, GAP scorer, required ingredient
  checkpoints, and MOGFN/HN-GFN comparators against the exact v1 contract.
- **Contract:** `β=32`; 10 preferences; 128 candidates per preference; top-10
  scalarized reward for ours, MOGFN, and HN-GFN.
- **Evidence:** [`evidence/claim-3/`](evidence/claim-3/), including the
  prerequisite checker and negative control.
- **Result:** the required QM9 ingredient and comparator checkpoints are absent;
  the included GAP scorer is not a generative checkpoint.
- **Assessment:** `BLOCKED`; no molecule proxy is counted.

## C4 — Logical molecule speed and accuracy

- **Anchor:** v1 Sections 6.2–6.3 and Tables 4–5.
- **Producer:** the same fail-closed molecule prerequisite audit checks the
  classifier-guidance implementation, checkpoints, timing path, target-bin
  thresholds, and imports.
- **Contract:** 1,000 timing samples after warmup and 5,000 target-bin samples
  for the required logical operators.
- **Evidence:** [`evidence/claim-4/`](evidence/claim-4/).
- **Result:** classifier guidance and timing/comparator surfaces are absent;
  the released QM9 SA/QED thresholds are `0.3`, while v1 specifies `0.4`.
- **Assessment:** `BLOCKED`; a different sampler or threshold would not be a
  faithful reproduction.

## C5 — Reaching-weight ablation

- **Anchor:** v1 Sections 5.1–5.2 and Table 1.
- **Producer:** the same content-addressed trained ingredients and 512 settings
  are evaluated after removing only `u_i(s)`, the reaching probability term.
- **Evidence:** [`evidence/claim-5/`](evidence/claim-5/), paired cells, and
  the removed-reaching mutation.
- **Contract:** paired improvement at least `0.05` in every seed/`k` cell and
  ensemble aggregates within `0.03` of the paper's range.
- **Result:** no-reaching ensemble means `0.1030–0.1239` versus ours
  `0.0071–0.0108`; every paired cell is worse by at least `0.05`.
- **Assessment:** `VERIFIED_SCOPED`.

## C6 — High-composition-value nonlinear distortion

- **Anchor:** v1 Section 5.3 and Figure 5, with the appendix stress audit
  reported separately.
- **Producer:** [`repro/src/run_claim_6_direct_audit.py`](repro/src/run_claim_6_direct_audit.py)
  directly records `G`, `u_M`, `N_M`, `δ=u_M/N_M`, induced mass, target mass,
  and L1 contribution for every state in the two primary operators over three
  seeds. The stdlib checker recomputes the identities and high/low-`G` stats.
- **Evidence:** [`evidence/claim-6/direct_state_audit.json`](evidence/claim-6/direct_state_audit.json),
  [`evidence/claim-6/direct_state_check.json`](evidence/claim-6/direct_state_check.json),
  and the shuffled-`N_M` negative control.
- **Contract:** six primary rows; high-`G` deviation, variance, RMSE, and error
  share must satisfy the declared comparisons. The broader 36-row appendix is
  descriptive, not a strengthened contract.
- **Result:** all six primary rows pass; the appendix has 31/36 rows with the
  same ordering and five reversals.
- **Assessment:** `VERIFIED_PRIMARY_SCOPE`.

## Limitations

The full neural output is replayed and checked from an immutable archive; it is
not silently regenerated in the cumulative node. Claims 3 and 4 cannot be
settled without the missing checkpoints/comparator code. The C6 positive result
is deliberately restricted to the paper-primary rows. Historical HF and blind
review scores are separate from these local claim verdicts.
