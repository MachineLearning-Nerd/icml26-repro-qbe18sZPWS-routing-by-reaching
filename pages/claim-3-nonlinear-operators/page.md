# Claim 3 — QM9 GAP-SA (legacy route mirror)

This retained route now mirrors the current official Claim 3. The historical
grid nonlinear-operator page is not evidence for the QM9 claim.

**Literal claim.** On atom-based QM9 GAP-SA, the method attains mean reward
`0.876`, exceeding MOGFN `0.816` and HN-GFN `0.805`. A faithful test requires
the three trained generation systems, ten preferences, 128 candidates per
preference, and the top-10 scalarized-reward aggregate.

**Result: BLOCKED — essential material unavailable.** The pinned author tree
contains `qm9.h5`, the MXMNet property scorer, and training code, but not the
trained GAP and SA ingredient GFlowNets or the MOGFN and HN-GFN comparators.
The property scorer is not a generative checkpoint. Recreating all missing
systems is a GPU training experiment, not a faithful two-hour CPU run.

The exact inventory is
[`molecule_prerequisite_audit.json`](../../evidence/claim-3/molecule_prerequisite_audit.json),
its independent check is
[`molecule_audit_checker.json`](../../evidence/claim-3/molecule_audit_checker.json),
and a mutation that pretends the checkpoints exist is rejected by
[`molecule_negative_control.json`](../../evidence/claim-3/molecule_negative_control.json).
No grid or scorer-only proxy is reported. See the complete
[Claim 3 — QM9 GAP-SA](#/claim-3-qm9-gap-sa) page.
