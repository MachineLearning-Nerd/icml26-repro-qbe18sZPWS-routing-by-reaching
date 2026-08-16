# Audit report

## Decision

This repository is a useful, reproducible archive with mixed outcomes:

```text
MIXED_RESULTS_WITH_LIMITATIONS
```

The exact linear composition contract (C1), the reaching-weight ablation (C5),
and the paper-primary nonlinear distortion contract (C6) have scoped positive
evidence. The exact printed neural C2 value is falsified by the source-scale
three-seed run, while its qualitative ordering remains strong. The molecule
claims C3 and C4 are blocked because the public source does not contain the
required checkpoints and comparator/timing surfaces.

## Why this is trustworthy

- The C2/C5 raw neural archive is content-addressed and evaluates all 48 models,
  512 grid settings, and 128 preferences per objective count.
- C6 directly records state-level quantities and has an independent stdlib
  checker plus a shuffled-state negative control.
- C3/C4 use fail-closed prerequisite audits; missing inputs remain `BLOCKED`.
- The five broader C6 reversals and the C2 rounding caveat are disclosed rather
  than hidden behind a single aggregate score.

## Boundary

Finite experiments do not prove universal GFlowNet theorems. This report does
not claim a new external judge score and does not treat a proxy molecule run as
evidence for C3 or C4.
