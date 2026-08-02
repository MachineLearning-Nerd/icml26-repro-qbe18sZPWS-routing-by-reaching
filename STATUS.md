# STATUS — Routing by Reaching (`qbe18sZPWS`)

**Last updated:** 2026-08-02. **State:** published — awaiting judge.

- Live judged baseline: `5/12` at HF `d34f1d6efa1b9c92cd9934c8a4a96218ea1c70a9`.
- Additive candidate: HF `8de344e3c4a45398170457c954d2b70dd62dab7c`.
- Space: `https://huggingface.co/spaces/DineshAI/qbe18sZPWS`.
- GitHub: `https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching`.

## Candidate evidence

- Claim 1: exact 512-setting full-state audit retained.
- Claim 2: full 48-model run exposes all 12 paired cells; the exact printed
  `0.003` is falsified while the qualitative advantage is reproduced.
- Claims 3–4: blocked by missing QM9 comparator/checkpoint and
  classifier-guidance/timing material.
- Claim 5: paired no-reaching ablation retained.
- Claim 6: six fresh 20,000-step checkpoints are byte-identical to the parent;
  direct `G`, `u_M`, `N_M`, `δ`, and induced-probability arrays cover all
  6,144 primary state/operator/seed combinations. All high-`G` variance,
  deviation, RMSE, and error-share comparisons pass; the shuffled-`N_M`
  control fails. A deterministic compact CSV now exposes the three largest-`G`
  terminals in every seed/operator row, including direct `u_M`, `N_M`, and
  `δ=u_M/N_M` values.

Post-publish verification matched all eight changed paths byte-for-byte, retained
all 126 baseline paths, and preserved the nine-node canonical page tree. The
new HEAD remains `awaiting judge`; no score increase is claimed.
