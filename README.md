# Repro — Routing by Reaching (ICML 2026)

Full-state reproduction of *Routing by Reaching: Composition of Pre-trained
GFlowNets for Multi-Objective Generation* ([arXiv:2602.21565](https://arxiv.org/abs/2602.21565),
OpenReview `qbe18sZPWS`).

## Results

1. **Training-free composition — verified.** Eight exact ingredient GFlowNets
   are composed in 524 unseen settings without any training. All computations
   use the paper's complete 32×32 HyperGrid (1,024 terminating states).
2. **Exact linear scalarization — verified.** Across 512 preference vectors and
   2–5 objectives, the largest L1 error is `8.32e-16`. An independent edge-flow
   conservation certificate also passes. Omitting reaching probabilities yields
   median L1 error `0.115`.
3. **Linear and nonlinear operators — verified with the stated qualification.**
   Harmonic-mean and contrast composition produce valid distributions in all 12
   paper-scale cases and enrich their intended regions. As the paper predicts,
   nonlinear operators are approximate (L1 `0.039–0.288`), while the derived
   distortion identity holds to roundoff.

Three fail-closed controls omit reaching weights, omit partition factors, or
apply the β=2 rule to a β=1 target. All are decisively rejected.

## Reproduce

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python repro/src/run_routing_by_reaching.py --output outputs/summary.json
pytest -q
```

Official source: `ml-postech/gflownet-composition`, pinned commit
`b82493c8cd9b46a0933ab8f19440aebbf3e14b28`.
