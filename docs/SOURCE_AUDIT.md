# Source and scope audit

The scored claims concern inference-time policy composition, exact recovery for
linear scalarization, and support for linear/nonlinear reward operators. The
paper's Section 5 synthetic evaluator enumerates a 32×32 HyperGrid, so this
reproduction uses that exact finite scale.

Instead of retraining approximate neural policies, we construct exact tabular
GFlowNets for the eight reward landscapes shipped by the authors. For each
reward, a uniform backward policy induces a valid state flow; its forward policy
terminates exactly in proportion to that reward. This is a stricter theorem test:
ingredient approximation cannot mask a composition error.

Verification is triangulated in two ways:

- forward probability rollout enumerates the induced terminal distribution;
- an independent flow audit checks every incoming, outgoing, and terminating
  edge against the weighted state-flow certificate.

For nonlinear operators, the paper does not claim literal exactness. We test
validity, intended concentration, and Equation (6)'s distortion identity, while
retaining the observed nonzero L1 errors in the output artifact.
