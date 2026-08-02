# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_rbr_conclusion_20260802", "created_at": "2026-08-02T11:49:48+00:00", "title": "Overall findings"}
-->
This full-scale CPU audit verifies Proposition 4.1 (Claim 1), the reaching-probability ablation (Claim 5), and the source-primary high-density distortion behavior (Claim 6). Claim 6 now includes direct `δ(x)=u_M(x)/N_M(x)` arrays over every primary state and six byte-identical checkpoint reruns, rather than only aggregate distortion identities. It falsifies Claim 2's exact printed `0.003` result while reproducing the method's qualitative advantage over both trained baselines in all 12 paired cells. Claims 3 and 4 remain `BLOCKED — essential material unavailable`: the public release lacks the required molecule ingredient/comparator checkpoints and classifier-guidance timing path, and its target-bin thresholds differ from arXiv v1. A blocked campaign is not called perfect.

All accepted evidence uses local CPU, fixed seeds, exact source pins, explicit controls, and deterministic checkers. The accepted neural record took 10 h 36 min; rerunning the cumulative release gate takes about 90 seconds with `uv sync --frozen && uv run python repro/src/run_campaign.py`. [Paper](https://arxiv.org/pdf/2602.21565v1) · [author source pin](https://github.com/ml-postech/gflownet-composition/tree/b82493c8cd9b46a0933ab8f19440aebbf3e14b28) · [reproduction repository](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching) · [Hugging Face Space](https://huggingface.co/spaces/DineshAI/qbe18sZPWS).

---
<!-- trackio-cell
{"type": "code", "id": "cell_rbr_release_gate", "created_at": "2026-08-02T11:49:48+00:00", "title": "Fail-closed cumulative gate", "command": ["uv", "run", "python", "repro/src/run_campaign.py"], "exit_code": 0, "duration_s": 88.11}
-->
````text title=cumulative_gate.txt
all six official claims present: PASS
exact cumulative verdicts retained: PASS
Claim 1 exact audit and three controls: PASS
Claim 2 source-scale archive, exact-point test, and mutation: PASS
Claims 3/4 prerequisite blockers and mutation: PASS
Claim 5 paired ablation and mutation: PASS
Claim 6 direct u_M/N_M state audit, broader stress audit, and mutation: PASS
full-grid raw copies content-addressed: PASS
cumulative verifier: PASS
````

The gate preserves negative outcomes: it cannot convert either molecule blocker into a success, cannot erase Claim 2's rounding limitation, and cannot extend Claim 6 beyond the source-primary scope. The fresh [v5 outcome-blind review](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/evidence/BLIND_REVIEW_V5.md) includes every retained claim route, scores the candidate evidence `8/12`—not a live leaderboard score—and records why Claims 3 and 4 prevent a perfect review.

The additive repair's complete [v3 campaign summary](https://huggingface.co/spaces/DineshAI/qbe18sZPWS/resolve/main/evidence/campaign_summary_v3.json) and [27-gate verifier output](https://huggingface.co/spaces/DineshAI/qbe18sZPWS/resolve/main/evidence/verifier_output_v3.json) preserve the scientific records above. The current live baseline is `4/12` at exact HF revision `8de344e3c4a45398170457c954d2b70dd62dab7c`; the route-consistency repair does not claim any unjudged points.
