import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Routing by Reaching: an evidence-first CPU reproduction

        ![Observed HyperGrid L1 errors and 95% confidence intervals](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/main/reports/routing-by-reaching-reproduction/images/headline-table1.png)

        This notebook explains the central claim without rerunning the 10.6-hour
        experiment. The plotted values and confidence intervals are embedded
        below from the content-addressed full-scale result.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The question

        A GFlowNet samples objects in proportion to a reward. The paper asks:
        if we already have one GFlowNet per objective, can we combine them at
        inference time instead of training a new multi-objective model?

        For a linear combination of rewards, the proposed policy routes through
        each ingredient using its objective weight *and* its probability of
        reaching the current state. The exact theorem says this recovers the
        desired terminal distribution. The empirical question is how accurately
        trained neural ingredients preserve that result.
        """
    )
    return


@app.cell
def _():
    results = {
        2: {
            "ours": 0.0108193369,
            "ci": (0.0050811073, 0.0165575665),
            "mogfn": 0.0271211498,
            "hngfn": 0.0278793328,
            "ensemble": 0.1238882963,
        },
        3: {
            "ours": 0.0079293789,
            "ci": (0.0041874249, 0.0116713329),
            "mogfn": 0.0315112610,
            "hngfn": 0.0239991711,
            "ensemble": 0.1049014049,
        },
        4: {
            "ours": 0.0072438913,
            "ci": (0.0034025751, 0.0110852075),
            "mogfn": 0.0429249895,
            "hngfn": 0.0390565189,
            "ensemble": 0.1180609540,
        },
        5: {
            "ours": 0.0071100431,
            "ci": (0.0055266670, 0.0086934192),
            "mogfn": 0.0555554513,
            "hngfn": 0.0411897305,
            "ensemble": 0.1030390650,
        },
    }
    return (results,)


@app.cell
def _(mo):
    objective_count = mo.ui.slider(
        start=2,
        stop=5,
        step=1,
        value=2,
        label="Number of objectives (k)",
    )
    objective_count
    return (objective_count,)


@app.cell
def _(mo, objective_count, results):
    row = results[objective_count.value]
    mo.md(
        f"""
        ### Embedded result for k={objective_count.value}

        | Method | Mean terminal L1 ↓ |
        |---|---:|
        | Routing by Reaching | **{row["ours"]:.5f}** |
        | MOGFN | {row["mogfn"]:.5f} |
        | HN-GFN | {row["hngfn"]:.5f} |
        | Without reaching weights | {row["ensemble"]:.5f} |

        The 95% t interval for the proposed method is
        **[{row["ci"][0]:.5f}, {row["ci"][1]:.5f}]**. The paper's exact
        Table 1 point is **0.003**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What the experiment establishes

        The test used the paper's 32×32 grid, 20,000 training steps, three
        deterministic seeds, and 128 preferences for each `k=2…5`. Forty-eight
        neural models were trained. Terminal probabilities were evaluated
        exactly over all 1,024 states.

        - The proposed method beats both trained baselines in every one of the
          12 seed-by-`k` cells.
        - The exact printed `0.003` point lies below all four two-sided 95% t
          intervals, so that narrow numerical claim is **FALSIFIED** under the
          machine contract.
        - A three-decimal rounding interval would overlap the `k=4` interval;
          the verdict does not erase that caveat.
        - Removing reaching weights worsens L1 by at least `0.05` in all 12
          paired cells, verifying the mechanism ablation.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Nonlinear composition

        For nonlinear operators the relevant diagnostic is

        \[
        \delta(x)=u_M(x)/N_M(x).
        \]

        On the two primary Figure 3 circle compositions, all six
        operator-by-seed rows have smaller relative distortion in the high-G
        decile than in the bottom half, and every high-G value is at most 0.30.
        The broader appendix stress test agrees in 31/36 rows—not 36/36—so the
        verified verdict is intentionally limited to the primary scope.

        ![High- versus low-G distortion in the primary Figure 3 settings](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/main/reports/routing-by-reaching-reproduction/images/distortion-primary.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Claim ledger

        | Claim | Verdict | Evidence |
        |---|---|---|
        | 1. Exact linear composition | **VERIFIED** | 512/512 settings; max L1 `8.61e-16` |
        | 2. Table 1 exact number | **FALSIFIED** | exact `0.003` below every 95% CI; ranking retained |
        | 3. QM9 GAP-SA | **BLOCKED** | required checkpoints and comparators unreleased |
        | 4. Molecule speed/accuracy | **BLOCKED** | classifier-guidance and timing surfaces absent |
        | 5. Reaching ablation | **VERIFIED** | all 12 paired gaps ≥ `0.05` |
        | 6. Primary distortion behavior | **VERIFIED** | six primary rows pass; 5/36 appendix reversals disclosed |

        The immutable raw JSON is identified by SHA-256
        `d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec`.
        See the [full illustrated report](https://github.com/MachineLearning-Nerd/icml26-repro-qbe18sZPWS-routing-by-reaching/blob/main/reports/routing-by-reaching-reproduction/report.md)
        for contracts, experiment lineage, blockers, and negative controls.
        """
    )
    return


if __name__ == "__main__":
    app.run()
