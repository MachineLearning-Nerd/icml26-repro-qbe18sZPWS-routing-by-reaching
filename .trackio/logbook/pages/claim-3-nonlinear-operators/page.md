# Claim 3 - Nonlinear operators


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f2b2044c34de", "created_at": "2026-07-17T04:04:02+00:00", "title": "Harmonic mean and contrast are evaluated on six reward pairs each. All induced…"}
-->
Harmonic mean and contrast are evaluated on six reward pairs each. All induced laws are normalized and nonnegative and enrich target-favored regions. Nonlinear composition is honestly approximate (L1 0.039–0.288); Equation (6)\u0027s distortion identity holds to roundoff.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9e4ea1de13ab", "created_at": "2026-07-19T15:12:00+00:00", "title": "Nonlinear operators on trained ingredients: 9/12 enriched, contrast 6/6"}
-->
## Diverse reward combinations on trained networks, without retraining

Harmonic-mean and contrast operators were applied to 6 pairs of the trained
ingredients (12 settings), producing composed samplers with zero retraining.
The composed distribution enriches the jointly-favored (top-decile operator
score) region versus BOTH ingredients in **9/12 settings — including 6/6 for
the contrast operator**. The 3 non-enriched settings are all harmonic pairs
involving the shubert ingredient (multi-peak; trained L1 0.115), where the
composed mass exceeds one ingredient's favored mass but not the other's —
reported verbatim. A 200,000-trajectory Monte-Carlo sample from a composed
policy matches its exact terminal distribution within 1.02x the analytic
multinomial noise floor, confirming the composed objects are genuine samplers.
