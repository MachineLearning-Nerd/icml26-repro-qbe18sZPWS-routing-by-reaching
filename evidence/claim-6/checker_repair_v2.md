# Claim 6 checker repair

The first source-scoped checker inherited the wrong figure number and introduced a `0.30` tolerance not stated by the paper. Outcome-blind review rejected both choices.

Version 2 anchors the audit to arXiv v1 Section 5.3, Figure 5, and Equation 9. Its predeclared checks use only direct paper-tied comparisons:

1. the high-`G` decile has lower median relative deviation than the bottom half in each of the six primary seed/operator rows;
2. the high-`G` decile's L1-error share is lower than its target-mass share in each row.

The immutable raw campaign is unchanged. The original checker files remain present for provenance, while the repaired contract, checker, control, source audit, and limitations use the `_v2` suffix.
