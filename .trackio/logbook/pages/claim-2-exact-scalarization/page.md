# Claim 2 - Exact scalarization


---
<!-- trackio-cell
{"type": "code", "id": "cell_26df49cbdb79", "created_at": "2026-07-17T04:04:00+00:00", "title": "Full 32x32 enumeration", "command": ["python", "repro/src/run_routing_by_reaching.py", "--output", "outputs/summary.json"], "exit_code": 0, "duration_s": 5.316}
-->
````bash
$ python repro/src/run_routing_by_reaching.py --output outputs/summary.json
````

exit 0 · 5.3s


````python title=run_routing_by_reaching.py
#!/usr/bin/env python3
"""Exact, full-HyperGrid reproduction of Routing by Reaching.

The paper evaluates a 32x32 DAG.  Here every state flow, reaching probability,
mixed transition, and terminal probability is enumerated in float64.  Ingredient
GFlowNets are exact tabular flows, which isolates the composition result from
neural approximation error while retaining the paper's complete state space.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def rewards(height: int = 32) -> dict[str, np.ndarray]:
    """Paper-code reward functions, translated from torch to NumPy."""
    a = np.arange(height, dtype=np.float64)
    x0, x1 = np.meshgrid(a, a, indexing="ij")
    unit0, unit1 = x0 / (height - 1), x1 / (height - 1)
    sym0, sym1 = 2 * unit0 - 1, 2 * unit1 - 1

    w, u = 2.3, -7.15
    sx0 = (u - w) + (np.abs(sym0) / 2 + 0.5) * w
    sx1 = (u - w) + (np.abs(sym1) / 2 + 0.5) * w
    c0 = sum(i * np.cos(sx0 * (i + 1) + i) for i in range(1, 6))
    c1 = sum(i * np.cos(sx1 * (i + 1) + i) for i in range(1, 6))
    shubert = np.maximum((c0 * c1 + 186.6157949555621) / 396.89241966352286, 0)

    factor = 1 - np.exp(-1 / (2 * unit1 + 1e-10))
    numer = 2300 * unit0**3 + 1900 * unit0**2 + 2092 * unit0 + 60
    denom = 100 * unit0**3 + 500 * unit0**2 + 4 * unit0 + 20
    currin = np.clip(factor * numer / denom / 13.77, 0, 1)

    bx, by = unit0 * 15 - 5, unit1 * 15
    bf = (by - 5.1 / (4 * np.pi**2) * bx**2 + 5 / np.pi * bx - 6) ** 2
    bf += 10 * (1 - 1 / (8 * np.pi)) * np.cos(bx) + 10
    branin = np.clip(1 - (bf - 0.397887) / (308.13 - 0.397887), 0, 1)

    scaled0, scaled1 = sym0 * 5.12, sym1 * 5.12
    sphere = (scaled0**2 + scaled1**2) / (2 * 5.12**2)
    diagonal = 1 / (1 + np.exp(-5 * (sym0 - sym1))) + 1e-5

    def circle(center, offsets, std, scale, bias):
        distance = np.sqrt((sym0 - center[0]) ** 2 + (sym1 - center[1]) ** 2)
        density = np.zeros_like(sym0)
        for ox, oy in offsets:
            density += np.exp(-0.5 * ((sym0 - center[0] - ox) ** 2 +
                                      (sym1 - center[1] - oy) ** 2) / std**2)
            density /= 1 if len(offsets) == 1 else 1  # averaging happens below
        density /= len(offsets) * 2 * np.pi * std**2
        return np.where(distance < 0.6, density * scale + bias, 0.1)

    h = 0.3
    c1r = circle((-h, 0), [(0, 0)], 0.3, 2.5, 6.5)
    hm = 0.32
    c2r = circle((h / 2, h * math.sqrt(3) / 2),
                 [(-hm * math.sqrt(3) / 2, hm / 2),
                  (hm * math.sqrt(3) / 2, -hm / 2)], 0.21, 3.5, 10.5)
    c3r = circle((h / 2, -h * math.sqrt(3) / 2),
                 [(-hm, 0), (hm / 2, math.sqrt(3) * hm / 2),
                  (hm / 2, -math.sqrt(3) * hm / 2)], 0.18, 2.5, 5.5)
    return {"shubert": shubert, "diagonal": diagonal, "currin": currin,
            "sphere": sphere, "branin": branin, "circle1": c1r,
            "circle2": c2r, "circle3": c3r}


@dataclass
class TabularGFN:
    reward: np.ndarray
    flow: np.ndarray
    policy: np.ndarray
    reaching: np.ndarray
    terminal: np.ndarray
    z: float


def exact_gfn(reward: np.ndarray) -> TabularGFN:
    """Build an exact flow with uniform backward policy on the grid DAG."""
    h = reward.shape[0]
    flow = np.zeros_like(reward)
    for i in range(h - 1, -1, -1):
        for j in range(h - 1, -1, -1):
            outgoing = 0.0
            if i + 1 < h:
                outgoing += flow[i + 1, j] / (1 + (j > 0))
            if j + 1 < h:
                outgoing += flow[i, j + 1] / (1 + (i > 0))
            flow[i, j] = reward[i, j] + outgoing
    policy = np.zeros((h, h, 3), dtype=np.float64)  # down, right, terminate
    for i in range(h):
        for j in range(h):
            if i + 1 < h:
                policy[i, j, 0] = flow[i + 1, j] / (1 + (j > 0)) / flow[i, j]
            if j + 1 < h:
                policy[i, j, 1] = flow[i, j + 1] / (1 + (i > 0)) / flow[i, j]
            policy[i, j, 2] = reward[i, j] / flow[i, j]
    reaching, terminal = rollout(policy)
    return TabularGFN(reward, flow, policy, reaching, terminal, float(flow[0, 0]))


def rollout(policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = policy.shape[0]
    u = np.zeros((h, h), dtype=np.float64)
    term = np.zeros((h, h), dtype=np.float64)
    u[0, 0] = 1.0
    for total in range(2 * h - 1):
        for i in range(max(0, total - h + 1), min(h, total + 1)):
            j = total - i
            if j >= h:
                continue
            term[i, j] = u[i, j] * policy[i, j, 2]
            if i + 1 < h:
                u[i + 1, j] += u[i, j] * policy[i, j, 0]
            if j + 1 < h:
                u[i, j + 1] += u[i, j] * policy[i, j, 1]
    return u, term


def scalar_mix(gfns: list[TabularGFN], weights: np.ndarray,
               use_reaching: bool = True) -> np.ndarray:
    if use_reaching:
        edge_mass = sum(w * g.z * g.reaching[..., None] * g.policy
                        for w, g in zip(weights, gfns))
    else:
        edge_mass = sum(w * g.z * g.policy for w, g in zip(weights, gfns))
    return edge_mass / edge_mass.sum(axis=-1, keepdims=True)


def operator(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "harmonic":
        prod = values.prod(axis=0)
        denom = sum(prod / np.maximum(values[i], 1e-300) for i in range(len(values)))
        return prod / np.maximum(denom, 1e-300)
    if kind == "contrast" and len(values) == 2:
        return values[0] ** 2 / np.maximum(values[0] + values[1], 1e-300)
    if kind == "contrast" and len(values) == 3:
        p1, p2, p3 = values
        return p1**4 / np.maximum((p1 + p2) * (p1**2 + p1*p3 + p2*p3), 1e-300)
    raise ValueError((kind, len(values)))


def nonlinear_mix(gfns: list[TabularGFN], kind: str,
                  use_reaching: bool = True) -> np.ndarray:
    values = np.stack([(g.reaching[..., None] if use_reaching else 1.0) * g.policy
                       for g in gfns])
    mass = operator(values, kind)
    return mass / mass.sum(axis=-1, keepdims=True)


def simplex_weights(k: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.dirichlet(np.ones(k), size=n)
    # Include boundaries and center to adversarially cover zero/symmetric weights.
    points[:k] = np.eye(k)
    points[k] = np.full(k, 1 / k)
    return points


def flow_conservation_residual(gfns: list[TabularGFN], weights: np.ndarray) -> float:
    """Independent edge-flow certificate; does not call the rollout evaluator."""
    h = gfns[0].reward.shape[0]
    node_flow = sum(w * g.flow for w, g in zip(weights, gfns))
    edge_flow = sum(w * g.flow[..., None] * g.policy for w, g in zip(weights, gfns))
    errors = [abs(edge_flow[0, 0].sum() - node_flow[0, 0])]
    for i in range(h):
        for j in range(h):
            errors.append(abs(edge_flow[i, j].sum() - node_flow[i, j]))
            if i or j:
                incoming = 0.0
                if i:
                    incoming += edge_flow[i - 1, j, 0]
                if j:
                    incoming += edge_flow[i, j - 1, 1]
                errors.append(abs(incoming - node_flow[i, j]))
            expected_stop = sum(w * g.reward[i, j] for w, g in zip(weights, gfns))
            errors.append(abs(edge_flow[i, j, 2] - expected_stop))
    return float(max(errors))


def main(output: Path) -> dict:
    start = time.perf_counter()
    rs = rewards(32)
    base_names = ["shubert", "diagonal", "currin", "sphere", "branin"]
    gfns = {name: exact_gfn(np.maximum(rs[name], 1e-12)) for name in rs}

    ingredient_checks = []
    for name, g in gfns.items():
        ingredient_checks.append({
            "name": name,
            "policy_sum_error": float(np.max(np.abs(g.policy.sum(-1) - 1))),
            "reaching_flow_error": float(np.max(np.abs(g.reaching - g.flow / g.z))),
            "terminal_target_l1": float(np.abs(g.terminal - g.reward / g.reward.sum()).sum()),
        })

    linear_rows = []
    for k in range(2, 6):
        chosen = [gfns[n] for n in base_names[:k]]
        for idx, w in enumerate(simplex_weights(k, 128, 100 + k)):
            policy = scalar_mix(chosen, w, True)
            _, induced = rollout(policy)
            target_reward = sum(wi * gi.reward for wi, gi in zip(w, chosen))
            target = target_reward / target_reward.sum()
            _, ablated = rollout(scalar_mix(chosen, w, False))
            linear_rows.append({"k": k, "preference": idx,
                                "l1": float(np.abs(induced - target).sum()),
                                "max_abs": float(np.max(np.abs(induced - target))),
                                "flow_certificate_residual": flow_conservation_residual(chosen, w),
                                "without_reaching_l1": float(np.abs(ablated - target).sum())})

    pairs = [("shubert", "sphere"), ("shubert", "diagonal"),
             ("branin", "sphere"), ("circle1", "circle2"),
             ("circle1", "circle3"), ("circle2", "circle3")]
    nonlinear_rows = []
    for kind in ("harmonic", "contrast"):
        for a, b in pairs:
            chosen = [gfns[a], gfns[b]]
            _, induced = rollout(nonlinear_mix(chosen, kind, True))
            mixed_policy = nonlinear_mix(chosen, kind, True)
            induced_reaching, induced_again = rollout(mixed_policy)
            _, ablated = rollout(nonlinear_mix(chosen, kind, False))
            probs = np.stack([g.terminal for g in chosen])
            target_mass = operator(probs, kind)
            target = target_mass / target_mass.sum()
            action_mass = operator(np.stack([g.reaching[..., None] * g.policy for g in chosen]), kind)
            local_norm = action_mass.sum(axis=-1)
            distortion_prediction = induced_reaching / local_norm * target_mass
            # Semantics: target-favored cells contain the top quartile of target mass.
            favored = target >= np.quantile(target, .75)
            nonlinear_rows.append({
                "operator": kind, "ingredients": [a, b],
                "l1": float(np.abs(induced - target).sum()),
                "distortion_identity_max_abs": float(np.max(np.abs(induced_again - distortion_prediction))),
                "without_reaching_l1": float(np.abs(ablated - target).sum()),
                "favored_mass_induced": float(induced[favored].sum()),
                "favored_mass_uniform": float(favored.mean()),
                "distribution_sum": float(induced.sum()),
                "distribution_min": float(induced.min()),
            })

    # Three falsifiers: each deliberately violates a theorem ingredient.
    chosen = [gfns["shubert"], gfns["diagonal"]]
    w = np.array([0.37, 0.63])
    target_reward = sum(wi * gi.reward for wi, gi in zip(w, chosen))
    target = target_reward / target_reward.sum()
    _, no_u = rollout(scalar_mix(chosen, w, False))
    wrong_z_mass = sum(wi * gi.reaching[..., None] * gi.policy for wi, gi in zip(w, chosen))
    _, wrong_z = rollout(wrong_z_mass / wrong_z_mass.sum(-1, keepdims=True))
    beta_mass = sum(wi * np.sqrt(gi.z * gi.reaching[..., None] * gi.policy)
                    for wi, gi in zip(w, chosen)) ** 2
    _, beta2 = rollout(beta_mass / beta_mass.sum(-1, keepdims=True))
    controls = {
        "omit_reaching_l1": float(np.abs(no_u - target).sum()),
        "omit_partition_l1": float(np.abs(wrong_z - target).sum()),
        "apply_beta2_to_beta1_target_l1": float(np.abs(beta2 - target).sum()),
    }

    elapsed = time.perf_counter() - start
    result = {
        "paper": "qbe18sZPWS",
        "scale": {"grid": "32x32", "states": 1024, "ingredients": 8,
                  "linear_settings": len(linear_rows),
                  "nonlinear_settings": len(nonlinear_rows)},
        "ingredient_checks": ingredient_checks,
        "claim_1": {
            "training_runs": 0, "adaptation_settings": len(linear_rows) + len(nonlinear_rows),
            "wall_seconds": elapsed,
            "all_distributions_valid": all(abs(r["distribution_sum"] - 1) < 1e-12 and
                                             r["distribution_min"] >= 0 for r in nonlinear_rows),
        },
        "claim_2": {
            "max_l1": max(r["l1"] for r in linear_rows),
            "max_abs": max(r["max_abs"] for r in linear_rows),
            "max_flow_certificate_residual": max(r["flow_certificate_residual"] for r in linear_rows),
            "median_without_reaching_l1": float(np.median([r["without_reaching_l1"] for r in linear_rows])),
            "settings": linear_rows,
        },
        "claim_3": {
            "operators": ["linear scalarization", "harmonic mean", "contrast"],
            "nonlinear": nonlinear_rows,
            "all_favored_enriched_over_uniform": all(r["favored_mass_induced"] > r["favored_mass_uniform"]
                                                      for r in nonlinear_rows),
        },
        "negative_controls": controls,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "scale": result["scale"], "wall_seconds": elapsed,
        "linear_max_l1": result["claim_2"]["max_l1"],
        "linear_max_abs": result["claim_2"]["max_abs"],
        "ablation_median_l1": result["claim_2"]["median_without_reaching_l1"],
        "nonlinear_l1_range": [min(r["l1"] for r in nonlinear_rows), max(r["l1"] for r in nonlinear_rows)],
        "controls": controls,
    }, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/summary.json"))
    args = parser.parse_args()
    main(args.output)

````


````json title=summary.json
{
  "paper": "qbe18sZPWS",
  "scale": {
    "grid": "32x32",
    "states": 1024,
    "ingredients": 8,
    "linear_settings": 512,
    "nonlinear_settings": 12
  },
  "ingredient_checks": [
    {
      "name": "shubert",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 5.551115123125783e-17,
      "terminal_target_l1": 1.9268983110498628e-16
    },
    {
      "name": "diagonal",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 1.1102230246251565e-16,
      "terminal_target_l1": 1.5848914208903e-16
    },
    {
      "name": "currin",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 1.1102230246251565e-16,
      "terminal_target_l1": 1.8238991046637398e-16
    },
    {
      "name": "sphere",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 5.551115123125783e-17,
      "terminal_target_l1": 1.2354949623593313e-16
    },
    {
      "name": "branin",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 5.551115123125783e-17,
      "terminal_target_l1": 3.148116541355029e-16
    },
    {
      "name": "circle1",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 1.1102230246251565e-16,
      "terminal_target_l1": 1.935233115250845e-16
    },
    {
      "name": "circle2",
      "policy_sum_error": 3.3306690738754696e-16,
      "reaching_flow_error": 1.1102230246251565e-16,
      "terminal_target_l1": 3.3588583303600927e-16
    },
    {
      "name": "circle3",
      "policy_sum_error": 2.220446049250313e-16,
      "reaching_flow_error": 1.1102230246251565e-16,
      "terminal_target_l1": 2.7639701508444525e-16
    }
  ],
  "claim_1": {
    "training_runs": 0,
    "adaptation_settings": 524,
    "wall_seconds": 5.186631512129679,
    "all_distributions_valid": true
  },
  "claim_2": {
    "max_l1": 8.317456966222547e-16,
    "max_abs": 4.336808689942018e-18,
    "max_flow_certificate_residual": 2.2737367544323206e-13,
    "median_without_reaching_l1": 0.11472514991085364,
    "settings": [
      {
        "k": 2,
        "preference": 0,
        "l1": 2.7021028643769984e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 7.105427357601002e-15,
        "without_reaching_l1": 2.5486882569702995e-16
      },
      {
        "k": 2,
        "preference": 1,
        "l1": 2.715071468195054e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.4210854715202004e-14,
        "without_reaching_l1": 1.9964393189727468e-16
      },
      {
        "k": 2,
        "preference": 2,
        "l1": 2.30846971312898e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1808060124295642
      },
      {
        "k": 2,
        "preference": 3,
        "l1": 2.8221104923439877e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18264990911376813
      },
      {
        "k": 2,
        "preference": 4,
        "l1": 1.964980912358416e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.01019982241391804
      },
      {
        "k": 2,
        "preference": 5,
        "l1": 4.218325721280086e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07164245097585309
      },
      {
        "k": 2,
        "preference": 6,
        "l1": 3.1490313286662575e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16366788952731234
      },
      {
        "k": 2,
        "preference": 7,
        "l1": 3.473919285915117e-16,
        "max_abs": 3.0357660829594124e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1829893947414687
      },
      {
        "k": 2,
        "preference": 8,
        "l1": 3.1665818513333666e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1606682153197996
      },
      {
        "k": 2,
        "preference": 9,
        "l1": 2.227527244689359e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1595976439130724
      },
      {
        "k": 2,
        "preference": 10,
        "l1": 2.8573640036087117e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07156476258350083
      },
      {
        "k": 2,
        "preference": 11,
        "l1": 3.002562391427044e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1318211271680116
      },
      {
        "k": 2,
        "preference": 12,
        "l1": 4.467272939642861e-16,
        "max_abs": 3.0357660829594124e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.024278389260649187
      },
      {
        "k": 2,
        "preference": 13,
        "l1": 3.015166241682188e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1832447956807854
      },
      {
        "k": 2,
        "preference": 14,
        "l1": 2.901528301478551e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18206586099218217
      },
      {
        "k": 2,
        "preference": 15,
        "l1": 4.459526823340221e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07357051733467129
      },
      {
        "k": 2,
        "preference": 16,
        "l1": 3.301395615218361e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16289977955208512
      },
      {
        "k": 2,
        "preference": 17,
        "l1": 2.6020852139652106e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16909820351106858
      },
      {
        "k": 2,
        "preference": 18,
        "l1": 2.682519462636479e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13941695001634813
      },
      {
        "k": 2,
        "preference": 19,
        "l1": 4.710282457045384e-16,
        "max_abs": 3.469446951953614e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1680974075049309
      },
      {
        "k": 2,
        "preference": 20,
        "l1": 3.0330216962103085e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10455782358295794
      },
      {
        "k": 2,
        "preference": 21,
        "l1": 3.174577842355447e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15018544346497978
      },
      {
        "k": 2,
        "preference": 22,
        "l1": 3.5843046196012973e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12112952415857128
      },
      {
        "k": 2,
        "preference": 23,
        "l1": 3.031903612719933e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18328934605318278
      },
      {
        "k": 2,
        "preference": 24,
        "l1": 2.7210764023954948e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11183527126222664
      },
      {
        "k": 2,
        "preference": 25,
        "l1": 3.6282825702227406e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11249853643369617
      },
      {
        "k": 2,
        "preference": 26,
        "l1": 2.900715149849187e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18033045005431353
      },
      {
        "k": 2,
        "preference": 27,
        "l1": 3.543037174411068e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13352681955238024
      },
      {
        "k": 2,
        "preference": 28,
        "l1": 3.0546718583421284e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14830635000001266
      },
      {
        "k": 2,
        "preference": 29,
        "l1": 2.246466901389965e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16569790465571596
      },
      {
        "k": 2,
        "preference": 30,
        "l1": 4.2239330793909097e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11079012948881835
      },
      {
        "k": 2,
        "preference": 31,
        "l1": 3.5382937899064437e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13237199150614193
      },
      {
        "k": 2,
        "preference": 32,
        "l1": 3.0245852480556557e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.051129383639097845
      },
      {
        "k": 2,
        "preference": 33,
        "l1": 3.295125453826311e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.004207576251117544
      },
      {
        "k": 2,
        "preference": 34,
        "l1": 3.51782947390078e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1251159105791207
      },
      {
        "k": 2,
        "preference": 35,
        "l1": 3.7117322561862343e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08052958239743185
      },
      {
        "k": 2,
        "preference": 36,
        "l1": 3.9113270998772376e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10898926469990758
      },
      {
        "k": 2,
        "preference": 37,
        "l1": 4.637064929084722e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.15242506689568058
      },
      {
        "k": 2,
        "preference": 38,
        "l1": 3.5004822391410118e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12203503096561999
      },
      {
        "k": 2,
        "preference": 39,
        "l1": 2.490242983609753e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15095500580688773
      },
      {
        "k": 2,
        "preference": 40,
        "l1": 4.0741607136574043e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0335049694143518
      },
      {
        "k": 2,
        "preference": 41,
        "l1": 5.305611093693596e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1175937638878772
      },
      {
        "k": 2,
        "preference": 42,
        "l1": 3.631128600925515e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14319434150680638
      },
      {
        "k": 2,
        "preference": 43,
        "l1": 3.706514533231148e-16,
        "max_abs": 3.469446951953614e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15334332806942644
      },
      {
        "k": 2,
        "preference": 44,
        "l1": 3.837533589512443e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06516284804783445
      },
      {
        "k": 2,
        "preference": 45,
        "l1": 2.3047935901378963e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08908380094891806
      },
      {
        "k": 2,
        "preference": 46,
        "l1": 2.493461708809319e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18438686542977015
      },
      {
        "k": 2,
        "preference": 47,
        "l1": 3.828046820503195e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10028944857649841
      },
      {
        "k": 2,
        "preference": 48,
        "l1": 5.48358965657067e-16,
        "max_abs": 3.2526065174565133e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16936828874992502
      },
      {
        "k": 2,
        "preference": 49,
        "l1": 4.956633519424825e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1293801020036724
      },
      {
        "k": 2,
        "preference": 50,
        "l1": 2.952350278313809e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18381102953778727
      },
      {
        "k": 2,
        "preference": 51,
        "l1": 3.702414893766437e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1780024996466219
      },
      {
        "k": 2,
        "preference": 52,
        "l1": 2.325613659981407e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.029868231877917796
      },
      {
        "k": 2,
        "preference": 53,
        "l1": 2.1515653299795934e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15858434340355032
      },
      {
        "k": 2,
        "preference": 54,
        "l1": 2.626615288117695e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 8.526512829121202e-14,
        "without_reaching_l1": 0.1846077374480024
      },
      {
        "k": 2,
        "preference": 55,
        "l1": 2.813301349692543e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0911396330074132
      },
      {
        "k": 2,
        "preference": 56,
        "l1": 3.3375808627250647e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.14985344666453693
      },
      {
        "k": 2,
        "preference": 57,
        "l1": 2.1852094786445342e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18444202362767545
      },
      {
        "k": 2,
        "preference": 58,
        "l1": 4.2310312154889007e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09078997815378882
      },
      {
        "k": 2,
        "preference": 59,
        "l1": 3.9494774638215713e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10217918684197796
      },
      {
        "k": 2,
        "preference": 60,
        "l1": 3.6282825702227406e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14841280940164742
      },
      {
        "k": 2,
        "preference": 61,
        "l1": 3.955711626313363e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11681404958046405
      },
      {
        "k": 2,
        "preference": 62,
        "l1": 3.133191812552602e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09785211793792468
      },
      {
        "k": 2,
        "preference": 63,
        "l1": 3.504886810466734e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1822517387420995
      },
      {
        "k": 2,
        "preference": 64,
        "l1": 2.7354420811809277e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17051615735700004
      },
      {
        "k": 2,
        "preference": 65,
        "l1": 3.129007469793166e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1743231424701097
      },
      {
        "k": 2,
        "preference": 66,
        "l1": 2.4068610602820395e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.01528525185901589
      },
      {
        "k": 2,
        "preference": 67,
        "l1": 4.2122609653777454e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1834258785530993
      },
      {
        "k": 2,
        "preference": 68,
        "l1": 2.5397774703651843e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08848556814792766
      },
      {
        "k": 2,
        "preference": 69,
        "l1": 3.3938916130585306e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09034504652701983
      },
      {
        "k": 2,
        "preference": 70,
        "l1": 3.950968241808739e-16,
        "max_abs": 3.0357660829594124e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08756515643680537
      },
      {
        "k": 2,
        "preference": 71,
        "l1": 3.0598556999793247e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14793788863214902
      },
      {
        "k": 2,
        "preference": 72,
        "l1": 2.1379789215056344e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18193401070728224
      },
      {
        "k": 2,
        "preference": 73,
        "l1": 3.532821957067181e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07735715121103268
      },
      {
        "k": 2,
        "preference": 74,
        "l1": 2.576775869501252e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.15772806551703117
      },
      {
        "k": 2,
        "preference": 75,
        "l1": 3.8337981742150515e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.022152282224138453
      },
      {
        "k": 2,
        "preference": 76,
        "l1": 4.2966254469242737e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0747055891130173
      },
      {
        "k": 2,
        "preference": 77,
        "l1": 2.582840625403593e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13897511476184954
      },
      {
        "k": 2,
        "preference": 78,
        "l1": 2.8445399247872816e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13629484695065233
      },
      {
        "k": 2,
        "preference": 79,
        "l1": 2.9134545253758914e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18138234419076893
      },
      {
        "k": 2,
        "preference": 80,
        "l1": 2.5217187279297226e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16986239272815642
      },
      {
        "k": 2,
        "preference": 81,
        "l1": 3.4606039279842793e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11762312398634285
      },
      {
        "k": 2,
        "preference": 82,
        "l1": 3.521607240845534e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06127710294353452
      },
      {
        "k": 2,
        "preference": 83,
        "l1": 2.882622526095835e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15003137537619776
      },
      {
        "k": 2,
        "preference": 84,
        "l1": 3.7870673665150317e-16,
        "max_abs": 3.469446951953614e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07994295498066861
      },
      {
        "k": 2,
        "preference": 85,
        "l1": 3.602397243354649e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16394889562737758
      },
      {
        "k": 2,
        "preference": 86,
        "l1": 3.727487069005164e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.16652262890497932
      },
      {
        "k": 2,
        "preference": 87,
        "l1": 2.880657409658205e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15030086168449314
      },
      {
        "k": 2,
        "preference": 88,
        "l1": 3.9818002410887954e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18264929843565933
      },
      {
        "k": 2,
        "preference": 89,
        "l1": 2.603101653501916e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12791948400523703
      },
      {
        "k": 2,
        "preference": 90,
        "l1": 4.013038816183534e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18148762794428275
      },
      {
        "k": 2,
        "preference": 91,
        "l1": 3.897495051848575e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.03676941573724285
      },
      {
        "k": 2,
        "preference": 92,
        "l1": 4.475857618563284e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07084952955703125
      },
      {
        "k": 2,
        "preference": 93,
        "l1": 2.5874484846366563e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1702507501054143
      },
      {
        "k": 2,
        "preference": 94,
        "l1": 2.916165030807105e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1841941864701176
      },
      {
        "k": 2,
        "preference": 95,
        "l1": 2.5738959574805875e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0845170230161377
      },
      {
        "k": 2,
        "preference": 96,
        "l1": 2.776031900013354e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13286979455419348
      },
      {
        "k": 2,
        "preference": 97,
        "l1": 2.5386593868748086e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1842011054875971
      },
      {
        "k": 2,
        "preference": 98,
        "l1": 2.8504352741001715e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1586047123203982
      },
      {
        "k": 2,
        "preference": 99,
        "l1": 4.652959502339949e-16,
        "max_abs": 3.686287386450715e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0038579456632796194
      },
      {
        "k": 2,
        "preference": 100,
        "l1": 4.214158319179595e-16,
        "max_abs": 4.336808689942018e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1830869899250821
      },
      {
        "k": 2,
        "preference": 101,
        "l1": 2.345264824357707e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16855620324065618
      },
      {
        "k": 2,
        "preference": 102,
        "l1": 3.4641614663627474e-16,
        "max_abs": 3.686287386450715e-18,
        "flow_certificate_residual": 8.526512829121202e-14,
        "without_reaching_l1": 0.04748268852781696
      },
      {
        "k": 2,
        "preference": 103,
        "l1": 3.784246746800675e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.032559418893927106
      },
      {
        "k": 2,
        "preference": 104,
        "l1": 3.4133733708453795e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.148416396866163
      },
      {
        "k": 2,
        "preference": 105,
        "l1": 2.419735961080305e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17248873476709048
      },
      {
        "k": 2,
        "preference": 106,
        "l1": 2.958042339719358e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06922910053833903
      },
      {
        "k": 2,
        "preference": 107,
        "l1": 4.67806132373183e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18335092342051773
      },
      {
        "k": 2,
        "preference": 108,
        "l1": 3.3182007488918863e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15178389437373355
      },
      {
        "k": 2,
        "preference": 109,
        "l1": 3.2153709490952143e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12500576643622835
      },
      {
        "k": 2,
        "preference": 110,
        "l1": 3.4072747336251485e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09682349866550242
      },
      {
        "k": 2,
        "preference": 111,
        "l1": 4.2349614483641607e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1446292395400571
      },
      {
        "k": 2,
        "preference": 112,
        "l1": 4.71939653155784e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18157862679072256
      },
      {
        "k": 2,
        "preference": 113,
        "l1": 4.448888089522707e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.014323499705771042
      },
      {
        "k": 2,
        "preference": 114,
        "l1": 3.0848262312643815e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16520261697153424
      },
      {
        "k": 2,
        "preference": 115,
        "l1": 4.740131898106625e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17580776225813155
      },
      {
        "k": 2,
        "preference": 116,
        "l1": 2.116430403327485e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18401466412898917
      },
      {
        "k": 2,
        "preference": 117,
        "l1": 2.8145888397723695e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.026525524951631577
      },
      {
        "k": 2,
        "preference": 118,
        "l1": 4.785160169582664e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11379952823522813
      },
      {
        "k": 2,
        "preference": 119,
        "l1": 3.183217578417441e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17505455632784553
      },
      {
        "k": 2,
        "preference": 120,
        "l1": 2.9215521603516426e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0625231679241425
      },
      {
        "k": 2,
        "preference": 121,
        "l1": 2.094576953288324e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17408907753340136
      },
      {
        "k": 2,
        "preference": 122,
        "l1": 4.323899907825862e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.17822522915706168
      },
      {
        "k": 2,
        "preference": 123,
        "l1": 2.5535671667464843e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18395740433371893
      },
      {
        "k": 2,
        "preference": 124,
        "l1": 3.4500160161436005e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.05997097749542954
      },
      {
        "k": 2,
        "preference": 125,
        "l1": 3.3249092498341404e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17446228758166063
      },
      {
        "k": 2,
        "preference": 126,
        "l1": 3.5317716362125856e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10937015098195858
      },
      {
        "k": 2,
        "preference": 127,
        "l1": 3.371496061933127e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14823165211505074
      },
      {
        "k": 3,
        "preference": 0,
        "l1": 2.7021028643769984e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 7.105427357601002e-15,
        "without_reaching_l1": 2.5486882569702995e-16
      },
      {
        "k": 3,
        "preference": 1,
        "l1": 2.715071468195054e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.4210854715202004e-14,
        "without_reaching_l1": 1.9964393189727468e-16
      },
      {
        "k": 3,
        "preference": 2,
        "l1": 3.4550812731681813e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 4.840420599061535e-16
      },
      {
        "k": 3,
        "preference": 3,
        "l1": 3.127245641262877e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13533327081421265
      },
      {
        "k": 3,
        "preference": 4,
        "l1": 3.3690227257271443e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13845024473389753
      },
      {
        "k": 3,
        "preference": 5,
        "l1": 4.036484688163533e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.05203007647569475
      },
      {
        "k": 3,
        "preference": 6,
        "l1": 3.5891157667417017e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15188921893968493
      },
      {
        "k": 3,
        "preference": 7,
        "l1": 2.636237582398504e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.03775459043342698
      },
      {
        "k": 3,
        "preference": 8,
        "l1": 3.6988234740700787e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1659565693524289
      },
      {
        "k": 3,
        "preference": 9,
        "l1": 6.289456802588411e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.004556546179645791
      },
      {
        "k": 3,
        "preference": 10,
        "l1": 2.8537556432534084e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1267625179145352
      },
      {
        "k": 3,
        "preference": 11,
        "l1": 3.0678855723192955e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10395806987166838
      },
      {
        "k": 3,
        "preference": 12,
        "l1": 2.7338157779221994e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07063188483392169
      },
      {
        "k": 3,
        "preference": 13,
        "l1": 2.6724228299052077e-16,
        "max_abs": 1.1926223897340549e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12346198382926822
      },
      {
        "k": 3,
        "preference": 14,
        "l1": 2.1787042656096212e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08263326140816028
      },
      {
        "k": 3,
        "preference": 15,
        "l1": 3.8226258096407673e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15214816732137487
      },
      {
        "k": 3,
        "preference": 16,
        "l1": 4.1730941618967066e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.019135430310003546
      },
      {
        "k": 3,
        "preference": 17,
        "l1": 4.3235272133290703e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06766269771133018
      },
      {
        "k": 3,
        "preference": 18,
        "l1": 4.0415330045291686e-16,
        "max_abs": 3.686287386450715e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07617500465046434
      },
      {
        "k": 3,
        "preference": 19,
        "l1": 3.355605723842636e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11085072230182716
      },
      {
        "k": 3,
        "preference": 20,
        "l1": 3.7300620491648173e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11540208968723473
      },
      {
        "k": 3,
        "preference": 21,
        "l1": 3.394365951508993e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.04423714105383692
      },
      {
        "k": 3,
        "preference": 22,
        "l1": 2.785721956929943e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 8.526512829121202e-14,
        "without_reaching_l1": 0.1171056265069122
      },
      {
        "k": 3,
        "preference": 23,
        "l1": 5.097783089755281e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1384459840832281
      },
      {
        "k": 3,
        "preference": 24,
        "l1": 2.5998490469844593e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08067443627413835
      },
      {
        "k": 3,
        "preference": 25,
        "l1": 2.945506252099994e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.01127182914336802
      },
      {
        "k": 3,
        "preference": 26,
        "l1": 4.775097418169283e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10632767539358008
      },
      {
        "k": 3,
        "preference": 27,
        "l1": 4.828629900435755e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12463458752132006
      },
      {
        "k": 3,
        "preference": 28,
        "l1": 2.7607853069627764e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.05096652458520077
      },
      {
        "k": 3,
        "preference": 29,
        "l1": 2.8967171543381465e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1311275957207749
      },
      {
        "k": 3,
        "preference": 30,
        "l1": 3.1618045855108523e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12340770064416451
      },
      {
        "k": 3,
        "preference": 31,
        "l1": 3.464839092720551e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04267058957399281
      },
      {
        "k": 3,
        "preference": 32,
        "l1": 3.1899938419954754e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06357791844459636
      },
      {
        "k": 3,
        "preference": 33,
        "l1": 3.001410426618778e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10683098074817994
      },
      {
        "k": 3,
        "preference": 34,
        "l1": 4.033503132189198e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07868334655257427
      },
      {
        "k": 3,
        "preference": 35,
        "l1": 5.11912832002609e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12387722934917662
      },
      {
        "k": 3,
        "preference": 36,
        "l1": 3.4592825565865626e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12216600004146863
      },
      {
        "k": 3,
        "preference": 37,
        "l1": 4.755310728521422e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09257137915866878
      },
      {
        "k": 3,
        "preference": 38,
        "l1": 2.2593079208703404e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03547313965567044
      },
      {
        "k": 3,
        "preference": 39,
        "l1": 3.3598070072610176e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07265495285061938
      },
      {
        "k": 3,
        "preference": 40,
        "l1": 2.481196671733077e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11862824785192704
      },
      {
        "k": 3,
        "preference": 41,
        "l1": 3.380948949624485e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1285765551518389
      },
      {
        "k": 3,
        "preference": 42,
        "l1": 2.6524328523500063e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1131585178280442
      },
      {
        "k": 3,
        "preference": 43,
        "l1": 2.8576858761286683e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.026690401896441004
      },
      {
        "k": 3,
        "preference": 44,
        "l1": 6.715683781646775e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1787881851050168
      },
      {
        "k": 3,
        "preference": 45,
        "l1": 2.4054380449306523e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13903704032642442
      },
      {
        "k": 3,
        "preference": 46,
        "l1": 2.499484113064297e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.02805012413980342
      },
      {
        "k": 3,
        "preference": 47,
        "l1": 2.4564633096732513e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1419422475342243
      },
      {
        "k": 3,
        "preference": 48,
        "l1": 2.3649159887340065e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13956235668014777
      },
      {
        "k": 3,
        "preference": 49,
        "l1": 2.8644621397067027e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09149919643923102
      },
      {
        "k": 3,
        "preference": 50,
        "l1": 2.1979488541712389e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06820170552164405
      },
      {
        "k": 3,
        "preference": 51,
        "l1": 5.835989243946349e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12861793580386158
      },
      {
        "k": 3,
        "preference": 52,
        "l1": 2.941304968681613e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13633537558760034
      },
      {
        "k": 3,
        "preference": 53,
        "l1": 3.2994982614165114e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.036540684689766595
      },
      {
        "k": 3,
        "preference": 54,
        "l1": 1.9821926218466235e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13441051413707428
      },
      {
        "k": 3,
        "preference": 55,
        "l1": 3.500888814955694e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11699877733601798
      },
      {
        "k": 3,
        "preference": 56,
        "l1": 2.9024092157436954e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15488659630342144
      },
      {
        "k": 3,
        "preference": 57,
        "l1": 2.387684234356202e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08172235436024243
      },
      {
        "k": 3,
        "preference": 58,
        "l1": 2.396899952822329e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.047083955603023524
      },
      {
        "k": 3,
        "preference": 59,
        "l1": 3.6014655071126694e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04966320255365163
      },
      {
        "k": 3,
        "preference": 60,
        "l1": 4.785126288264774e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11241621439095473
      },
      {
        "k": 3,
        "preference": 61,
        "l1": 4.4001667543966394e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11674665537897308
      },
      {
        "k": 3,
        "preference": 62,
        "l1": 2.998225582737102e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12089808241131353
      },
      {
        "k": 3,
        "preference": 63,
        "l1": 3.562417288244246e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06488791753586393
      },
      {
        "k": 3,
        "preference": 64,
        "l1": 4.545246557602356e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06191758411056614
      },
      {
        "k": 3,
        "preference": 65,
        "l1": 2.596664203102783e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04691758318760027
      },
      {
        "k": 3,
        "preference": 66,
        "l1": 3.1433731485785987e-16,
        "max_abs": 9.75781955236954e-19,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0991044888660894
      },
      {
        "k": 3,
        "preference": 67,
        "l1": 4.714788672324777e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11950143436845187
      },
      {
        "k": 3,
        "preference": 68,
        "l1": 3.6030071070766723e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09096022534002927
      },
      {
        "k": 3,
        "preference": 69,
        "l1": 2.472387529081632e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13793630359763098
      },
      {
        "k": 3,
        "preference": 70,
        "l1": 2.8284124174715597e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14145272045295693
      },
      {
        "k": 3,
        "preference": 71,
        "l1": 2.6195679739965394e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11337704631134875
      },
      {
        "k": 3,
        "preference": 72,
        "l1": 3.1891806903661113e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10019701702397218
      },
      {
        "k": 3,
        "preference": 73,
        "l1": 4.825936335663486e-16,
        "max_abs": 3.2526065174565133e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03759444192493291
      },
      {
        "k": 3,
        "preference": 74,
        "l1": 3.2194028259241447e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08446630938136222
      },
      {
        "k": 3,
        "preference": 75,
        "l1": 3.4488471106763896e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03125459218151796
      },
      {
        "k": 3,
        "preference": 76,
        "l1": 2.884384354626124e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16954740459333514
      },
      {
        "k": 3,
        "preference": 77,
        "l1": 4.162523190714973e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.031713316509542705
      },
      {
        "k": 3,
        "preference": 78,
        "l1": 2.2475511035624507e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09331638898807729
      },
      {
        "k": 3,
        "preference": 79,
        "l1": 1.9931701688430392e-16,
        "max_abs": 6.505213034913027e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1296197811711023
      },
      {
        "k": 3,
        "preference": 80,
        "l1": 3.3015311404899217e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15779650604639447
      },
      {
        "k": 3,
        "preference": 81,
        "l1": 5.228700502082906e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1127607322378043
      },
      {
        "k": 3,
        "preference": 82,
        "l1": 3.1786097191843776e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.022547881201526267
      },
      {
        "k": 3,
        "preference": 83,
        "l1": 3.1173522964389466e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11826845745825866
      },
      {
        "k": 3,
        "preference": 84,
        "l1": 3.9046185989349835e-16,
        "max_abs": 3.0357660829594124e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14411698308314105
      },
      {
        "k": 3,
        "preference": 85,
        "l1": 2.9069493123409784e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10645017277173202
      },
      {
        "k": 3,
        "preference": 86,
        "l1": 5.281657001945245e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11932997054399211
      },
      {
        "k": 3,
        "preference": 87,
        "l1": 2.704948895079773e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09507659126461838
      },
      {
        "k": 3,
        "preference": 88,
        "l1": 3.4714798310270245e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11404672051180192
      },
      {
        "k": 3,
        "preference": 89,
        "l1": 3.8555584506300145e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09647450676905521
      },
      {
        "k": 3,
        "preference": 90,
        "l1": 3.101631364937907e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1475796340001299
      },
      {
        "k": 3,
        "preference": 91,
        "l1": 2.5977484052752686e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15178872087904322
      },
      {
        "k": 3,
        "preference": 92,
        "l1": 3.2692083632226976e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16170305556371067
      },
      {
        "k": 3,
        "preference": 93,
        "l1": 2.0942720214273125e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1317042017223471
      },
      {
        "k": 3,
        "preference": 94,
        "l1": 2.9227380064777986e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08970909757066223
      },
      {
        "k": 3,
        "preference": 95,
        "l1": 3.623674710989677e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07440291983732428
      },
      {
        "k": 3,
        "preference": 96,
        "l1": 3.9302328752599536e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07359732948578167
      },
      {
        "k": 3,
        "preference": 97,
        "l1": 3.2764589652511944e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13171174225078291
      },
      {
        "k": 3,
        "preference": 98,
        "l1": 4.472333961502706e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.023865310529286497
      },
      {
        "k": 3,
        "preference": 99,
        "l1": 4.691342800344778e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.01517831682985412
      },
      {
        "k": 3,
        "preference": 100,
        "l1": 2.9926690466031136e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.061615168720493406
      },
      {
        "k": 3,
        "preference": 101,
        "l1": 3.5865407865820487e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14397060043862264
      },
      {
        "k": 3,
        "preference": 102,
        "l1": 3.5309754252421666e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06975506139173107
      },
      {
        "k": 3,
        "preference": 103,
        "l1": 3.8677557250704764e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1155877390651014
      },
      {
        "k": 3,
        "preference": 104,
        "l1": 5.608374550360173e-16,
        "max_abs": 3.469446951953614e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12983536345780317
      },
      {
        "k": 3,
        "preference": 105,
        "l1": 4.573300288815418e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.15327863335808456
      },
      {
        "k": 3,
        "preference": 106,
        "l1": 3.7343988578547593e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14559730334328558
      },
      {
        "k": 3,
        "preference": 107,
        "l1": 3.0196385756436905e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09249762530570971
      },
      {
        "k": 3,
        "preference": 108,
        "l1": 5.455705331947058e-16,
        "max_abs": 3.686287386450715e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11310016975776255
      },
      {
        "k": 3,
        "preference": 109,
        "l1": 5.359753439682091e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.126088537243559
      },
      {
        "k": 3,
        "preference": 110,
        "l1": 2.5031856470437985e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09273210126221934
      },
      {
        "k": 3,
        "preference": 111,
        "l1": 3.412255287355004e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0915143247770434
      },
      {
        "k": 3,
        "preference": 112,
        "l1": 1.913345783893794e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.035078906221287606
      },
      {
        "k": 3,
        "preference": 113,
        "l1": 3.47256403319951e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1487643244105146
      },
      {
        "k": 3,
        "preference": 114,
        "l1": 2.2441629717734335e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1033634416868659
      },
      {
        "k": 3,
        "preference": 115,
        "l1": 3.563365965145171e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.08450240628287742
      },
      {
        "k": 3,
        "preference": 116,
        "l1": 3.7848142588753353e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08619733746820384
      },
      {
        "k": 3,
        "preference": 117,
        "l1": 3.7894221181083987e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07727557537578429
      },
      {
        "k": 3,
        "preference": 118,
        "l1": 3.2337685047095777e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10892979598750502
      },
      {
        "k": 3,
        "preference": 119,
        "l1": 3.8685688766998405e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11739167754442312
      },
      {
        "k": 3,
        "preference": 120,
        "l1": 2.836543933765201e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.02883674343388916
      },
      {
        "k": 3,
        "preference": 121,
        "l1": 2.4623586589861413e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06083563426662483
      },
      {
        "k": 3,
        "preference": 122,
        "l1": 4.235571312086184e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11569869516216719
      },
      {
        "k": 3,
        "preference": 123,
        "l1": 6.417121608398579e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10864157258600782
      },
      {
        "k": 3,
        "preference": 124,
        "l1": 2.4245471082207093e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07524330863805398
      },
      {
        "k": 3,
        "preference": 125,
        "l1": 2.5837215396687374e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1715134236029957
      },
      {
        "k": 3,
        "preference": 126,
        "l1": 3.65376132127615e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.03811310494889819
      },
      {
        "k": 3,
        "preference": 127,
        "l1": 4.81792340398246e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07412646252580227
      },
      {
        "k": 4,
        "preference": 0,
        "l1": 2.7021028643769984e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 7.105427357601002e-15,
        "without_reaching_l1": 2.5486882569702995e-16
      },
      {
        "k": 4,
        "preference": 1,
        "l1": 2.715071468195054e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.4210854715202004e-14,
        "without_reaching_l1": 1.9964393189727468e-16
      },
      {
        "k": 4,
        "preference": 2,
        "l1": 3.4550812731681813e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 4.840420599061535e-16
      },
      {
        "k": 4,
        "preference": 3,
        "l1": 4.278130482520976e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.7763568394002505e-15,
        "without_reaching_l1": 3.321017133441507e-16
      },
      {
        "k": 4,
        "preference": 4,
        "l1": 1.966742740888705e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1409498127042026
      },
      {
        "k": 4,
        "preference": 5,
        "l1": 2.222614453595284e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04677398633140717
      },
      {
        "k": 4,
        "preference": 6,
        "l1": 2.268693045925918e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17092050241975693
      },
      {
        "k": 4,
        "preference": 7,
        "l1": 5.695042961523233e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12763153694587648
      },
      {
        "k": 4,
        "preference": 8,
        "l1": 2.8850619809839273e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1509752762168517
      },
      {
        "k": 4,
        "preference": 9,
        "l1": 3.188096488193626e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0652929445725556
      },
      {
        "k": 4,
        "preference": 10,
        "l1": 2.611571982974459e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13584445155776498
      },
      {
        "k": 4,
        "preference": 11,
        "l1": 2.1581044243323966e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13874761957512188
      },
      {
        "k": 4,
        "preference": 12,
        "l1": 2.4492127076447545e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13852340874008617
      },
      {
        "k": 4,
        "preference": 13,
        "l1": 2.8454886016882064e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.049700670515196846
      },
      {
        "k": 4,
        "preference": 14,
        "l1": 7.434645347276225e-16,
        "max_abs": 3.0357660829594124e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1368641753030175
      },
      {
        "k": 4,
        "preference": 15,
        "l1": 2.9322247754870467e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09073943786409486
      },
      {
        "k": 4,
        "preference": 16,
        "l1": 3.6440035017237804e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1220775568056095
      },
      {
        "k": 4,
        "preference": 17,
        "l1": 2.9127091363823077e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12921268994822394
      },
      {
        "k": 4,
        "preference": 18,
        "l1": 3.920475055707584e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1275754823973621
      },
      {
        "k": 4,
        "preference": 19,
        "l1": 2.8937355983638113e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1808514328953284
      },
      {
        "k": 4,
        "preference": 20,
        "l1": 2.8224493055228894e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.044433971595227356
      },
      {
        "k": 4,
        "preference": 21,
        "l1": 3.789286592836838e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12503651164571816
      },
      {
        "k": 4,
        "preference": 22,
        "l1": 2.5256489608049826e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13366740715894157
      },
      {
        "k": 4,
        "preference": 23,
        "l1": 3.0558238231503942e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08998186507028039
      },
      {
        "k": 4,
        "preference": 24,
        "l1": 3.201106914263452e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.054832494054781314
      },
      {
        "k": 4,
        "preference": 25,
        "l1": 2.4990860075790877e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07197671190972721
      },
      {
        "k": 4,
        "preference": 26,
        "l1": 3.377831868378589e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03678590083936206
      },
      {
        "k": 4,
        "preference": 27,
        "l1": 4.821447061043038e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17482224802967997
      },
      {
        "k": 4,
        "preference": 28,
        "l1": 2.467644144577008e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10165822277835886
      },
      {
        "k": 4,
        "preference": 29,
        "l1": 2.4362022815749285e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14720448381029372
      },
      {
        "k": 4,
        "preference": 30,
        "l1": 2.9463194037293583e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09444859405728115
      },
      {
        "k": 4,
        "preference": 31,
        "l1": 3.1870122860211403e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12329038386004357
      },
      {
        "k": 4,
        "preference": 32,
        "l1": 2.8957684774372217e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.17239054697861939
      },
      {
        "k": 4,
        "preference": 33,
        "l1": 4.856954682191938e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15531976861914565
      },
      {
        "k": 4,
        "preference": 34,
        "l1": 4.2500725161431774e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10461165605569821
      },
      {
        "k": 4,
        "preference": 35,
        "l1": 2.1515992112974835e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15436466916039326
      },
      {
        "k": 4,
        "preference": 36,
        "l1": 4.0310636773011055e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1046913123721378
      },
      {
        "k": 4,
        "preference": 37,
        "l1": 4.0706370565968264e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13676961635425583
      },
      {
        "k": 4,
        "preference": 38,
        "l1": 2.26706674266719e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.12312302873356523
      },
      {
        "k": 4,
        "preference": 39,
        "l1": 2.1927988938519327e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09187921667536847
      },
      {
        "k": 4,
        "preference": 40,
        "l1": 3.168038748002644e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08945629478533507
      },
      {
        "k": 4,
        "preference": 41,
        "l1": 4.05166351857833e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1785303517133928
      },
      {
        "k": 4,
        "preference": 42,
        "l1": 3.073713158996405e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13387786360748194
      },
      {
        "k": 4,
        "preference": 43,
        "l1": 2.1586465254186393e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06089226839170201
      },
      {
        "k": 4,
        "preference": 44,
        "l1": 2.137504583055172e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14378622883387254
      },
      {
        "k": 4,
        "preference": 45,
        "l1": 2.6150956400350367e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1522075510639223
      },
      {
        "k": 4,
        "preference": 46,
        "l1": 5.928417479150738e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11976846003266138
      },
      {
        "k": 4,
        "preference": 47,
        "l1": 2.6194324487249787e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11142164215962459
      },
      {
        "k": 4,
        "preference": 48,
        "l1": 3.2477276076803285e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16079329289353092
      },
      {
        "k": 4,
        "preference": 49,
        "l1": 2.680147770384167e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07327058852698867
      },
      {
        "k": 4,
        "preference": 50,
        "l1": 2.4287483916390906e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1251396993545625
      },
      {
        "k": 4,
        "preference": 51,
        "l1": 3.2124910370745496e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14877340627738073
      },
      {
        "k": 4,
        "preference": 52,
        "l1": 2.3787395664331967e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.05672201237982259
      },
      {
        "k": 4,
        "preference": 53,
        "l1": 2.7647155398380363e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.08546200329394753
      },
      {
        "k": 4,
        "preference": 54,
        "l1": 2.4226497544188597e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11855961655325988
      },
      {
        "k": 4,
        "preference": 55,
        "l1": 2.1147363374329764e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13538273337942722
      },
      {
        "k": 4,
        "preference": 56,
        "l1": 2.9024092157436954e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06399356606919751
      },
      {
        "k": 4,
        "preference": 57,
        "l1": 3.2100515821864573e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.05029250674778045
      },
      {
        "k": 4,
        "preference": 58,
        "l1": 2.9783033678176807e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.166546652013853
      },
      {
        "k": 4,
        "preference": 59,
        "l1": 3.1013603143947854e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09922992159604556
      },
      {
        "k": 4,
        "preference": 60,
        "l1": 2.3006770100142404e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.7053025658242404e-13,
        "without_reaching_l1": 0.13295807563119105
      },
      {
        "k": 4,
        "preference": 61,
        "l1": 3.4027685183457557e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15287185702309813
      },
      {
        "k": 4,
        "preference": 62,
        "l1": 2.1719280020315868e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09523352596579274
      },
      {
        "k": 4,
        "preference": 63,
        "l1": 2.2724877535296173e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1450630218875575
      },
      {
        "k": 4,
        "preference": 64,
        "l1": 3.1192496502407963e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13309223875413081
      },
      {
        "k": 4,
        "preference": 65,
        "l1": 5.324245818533191e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14973757628190557
      },
      {
        "k": 4,
        "preference": 66,
        "l1": 2.6546690193307576e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.11666032356722306
      },
      {
        "k": 4,
        "preference": 67,
        "l1": 4.033503132189198e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.16180860662243363
      },
      {
        "k": 4,
        "preference": 68,
        "l1": 3.067750047047735e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.01907714749581007
      },
      {
        "k": 4,
        "preference": 69,
        "l1": 4.0093796338513954e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11572615805112843
      },
      {
        "k": 4,
        "preference": 70,
        "l1": 2.5348646792711094e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15984649441167395
      },
      {
        "k": 4,
        "preference": 71,
        "l1": 2.5738959574805875e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13616006109121662
      },
      {
        "k": 4,
        "preference": 72,
        "l1": 3.120333852413282e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16207177352747643
      },
      {
        "k": 4,
        "preference": 73,
        "l1": 3.364821442308763e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0896082403027422
      },
      {
        "k": 4,
        "preference": 74,
        "l1": 5.367884955975732e-16,
        "max_abs": 3.2526065174565133e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10715633937677102
      },
      {
        "k": 4,
        "preference": 75,
        "l1": 5.396616313546598e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.16610150575580285
      },
      {
        "k": 4,
        "preference": 76,
        "l1": 2.8303097712734093e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09269006722095748
      },
      {
        "k": 4,
        "preference": 77,
        "l1": 2.6703899508317974e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16641727584828137
      },
      {
        "k": 4,
        "preference": 78,
        "l1": 2.0664893407573715e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.10377026541283141
      },
      {
        "k": 4,
        "preference": 79,
        "l1": 2.992940097146235e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14367710692671412
      },
      {
        "k": 4,
        "preference": 80,
        "l1": 2.4874308342248685e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0853202546978313
      },
      {
        "k": 4,
        "preference": 81,
        "l1": 2.667137344314341e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.19513343757702545
      },
      {
        "k": 4,
        "preference": 82,
        "l1": 6.694270788740186e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11561157257967
      },
      {
        "k": 4,
        "preference": 83,
        "l1": 2.443791696782327e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1297303447286477
      },
      {
        "k": 4,
        "preference": 84,
        "l1": 3.9581510812014553e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06837142052665543
      },
      {
        "k": 4,
        "preference": 85,
        "l1": 3.900959416602845e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09582661421240563
      },
      {
        "k": 4,
        "preference": 86,
        "l1": 2.355971320811001e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12523445551584728
      },
      {
        "k": 4,
        "preference": 87,
        "l1": 4.888125494650897e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15992833356856517
      },
      {
        "k": 4,
        "preference": 88,
        "l1": 4.2256779672622535e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08026779360028027
      },
      {
        "k": 4,
        "preference": 89,
        "l1": 3.862741290022731e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09691939192189802
      },
      {
        "k": 4,
        "preference": 90,
        "l1": 3.247185506594086e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12077782930342014
      },
      {
        "k": 4,
        "preference": 91,
        "l1": 4.434386885465713e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.053276757272513464
      },
      {
        "k": 4,
        "preference": 92,
        "l1": 2.1423834928313568e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.084309591481212
      },
      {
        "k": 4,
        "preference": 93,
        "l1": 3.139849491518021e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07274995848309768
      },
      {
        "k": 4,
        "preference": 94,
        "l1": 3.102444516567271e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09504031119082515
      },
      {
        "k": 4,
        "preference": 95,
        "l1": 4.343313902976931e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16638548516706855
      },
      {
        "k": 4,
        "preference": 96,
        "l1": 2.8441333489725995e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13849388443026708
      },
      {
        "k": 4,
        "preference": 97,
        "l1": 4.0107348865670023e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04937591767365541
      },
      {
        "k": 4,
        "preference": 98,
        "l1": 2.7901942908914457e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1741052030794183
      },
      {
        "k": 4,
        "preference": 99,
        "l1": 2.8376281359376865e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15880973957107555
      },
      {
        "k": 4,
        "preference": 100,
        "l1": 3.3664477455674913e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07935175712602946
      },
      {
        "k": 4,
        "preference": 101,
        "l1": 3.565940945304824e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14573394265326395
      },
      {
        "k": 4,
        "preference": 102,
        "l1": 2.830851872359652e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13138348996245236
      },
      {
        "k": 4,
        "preference": 103,
        "l1": 2.857414825585547e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0727043180961005
      },
      {
        "k": 4,
        "preference": 104,
        "l1": 4.522749362523282e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1387136249738309
      },
      {
        "k": 4,
        "preference": 105,
        "l1": 2.1662359406260379e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1708232679476766
      },
      {
        "k": 4,
        "preference": 106,
        "l1": 2.8075415256512137e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.060259854092068806
      },
      {
        "k": 4,
        "preference": 107,
        "l1": 2.900782912484967e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07467481346264364
      },
      {
        "k": 4,
        "preference": 108,
        "l1": 2.983724378680108e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 8.526512829121202e-14,
        "without_reaching_l1": 0.09622731758446687
      },
      {
        "k": 4,
        "preference": 109,
        "l1": 2.238335385096324e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.17795520089877828
      },
      {
        "k": 4,
        "preference": 110,
        "l1": 4.395762183070917e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14774362665953092
      },
      {
        "k": 4,
        "preference": 111,
        "l1": 2.9934821982324777e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15410263483942807
      },
      {
        "k": 4,
        "preference": 112,
        "l1": 2.9436088982981445e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09106833399030453
      },
      {
        "k": 4,
        "preference": 113,
        "l1": 2.956890374911092e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13028600872659696
      },
      {
        "k": 4,
        "preference": 114,
        "l1": 3.990948196919142e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12848199687959586
      },
      {
        "k": 4,
        "preference": 115,
        "l1": 4.657190431911484e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10013928526515331
      },
      {
        "k": 4,
        "preference": 116,
        "l1": 2.359088402056897e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08225372588312878
      },
      {
        "k": 4,
        "preference": 117,
        "l1": 4.601896121114724e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13823143791531373
      },
      {
        "k": 4,
        "preference": 118,
        "l1": 2.6503322106408156e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1765009368603402
      },
      {
        "k": 4,
        "preference": 119,
        "l1": 2.9089144287786084e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09084895770289181
      },
      {
        "k": 4,
        "preference": 120,
        "l1": 2.379010616976318e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09334237365597473
      },
      {
        "k": 4,
        "preference": 121,
        "l1": 3.8846963840155624e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.13493416762843038
      },
      {
        "k": 4,
        "preference": 122,
        "l1": 4.643095803669173e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0927737505879905
      },
      {
        "k": 4,
        "preference": 123,
        "l1": 2.875575211974679e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04207732088626562
      },
      {
        "k": 4,
        "preference": 124,
        "l1": 2.516433242338856e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16923271909831664
      },
      {
        "k": 4,
        "preference": 125,
        "l1": 4.782957883919803e-16,
        "max_abs": 3.0357660829594124e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14594633344398172
      },
      {
        "k": 4,
        "preference": 126,
        "l1": 4.1649626456030653e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1453195044518147
      },
      {
        "k": 4,
        "preference": 127,
        "l1": 3.4380050889515346e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1975185081032243
      },
      {
        "k": 5,
        "preference": 0,
        "l1": 2.7021028643769984e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 7.105427357601002e-15,
        "without_reaching_l1": 2.5486882569702995e-16
      },
      {
        "k": 5,
        "preference": 1,
        "l1": 2.715071468195054e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.4210854715202004e-14,
        "without_reaching_l1": 1.9964393189727468e-16
      },
      {
        "k": 5,
        "preference": 2,
        "l1": 3.4550812731681813e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 4.840420599061535e-16
      },
      {
        "k": 5,
        "preference": 3,
        "l1": 4.278130482520976e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.7763568394002505e-15,
        "without_reaching_l1": 3.321017133441507e-16
      },
      {
        "k": 5,
        "preference": 4,
        "l1": 3.0006650458970003e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 7.105427357601002e-15,
        "without_reaching_l1": 2.7660708008254493e-16
      },
      {
        "k": 5,
        "preference": 5,
        "l1": 4.840962700147777e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1162270132639561
      },
      {
        "k": 5,
        "preference": 6,
        "l1": 4.145447006498326e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.165539596461793
      },
      {
        "k": 5,
        "preference": 7,
        "l1": 4.635506388461774e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11270293281347046
      },
      {
        "k": 5,
        "preference": 8,
        "l1": 2.811336233254913e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1453718143462759
      },
      {
        "k": 5,
        "preference": 9,
        "l1": 1.9607796289400348e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.038574609662785156
      },
      {
        "k": 5,
        "preference": 10,
        "l1": 3.1403915926042636e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.058272726300271346
      },
      {
        "k": 5,
        "preference": 11,
        "l1": 4.734168786157955e-16,
        "max_abs": 3.2526065174565133e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07612089087899497
      },
      {
        "k": 5,
        "preference": 12,
        "l1": 2.706168622523819e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16728922758033649
      },
      {
        "k": 5,
        "preference": 13,
        "l1": 2.1575623232461538e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0507283889672329
      },
      {
        "k": 5,
        "preference": 14,
        "l1": 2.2123145329566718e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1488021581954212
      },
      {
        "k": 5,
        "preference": 15,
        "l1": 2.553296116203363e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12406133719225407
      },
      {
        "k": 5,
        "preference": 16,
        "l1": 2.2350827785788674e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10742397079070888
      },
      {
        "k": 5,
        "preference": 17,
        "l1": 4.531422979903166e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06844684882410344
      },
      {
        "k": 5,
        "preference": 18,
        "l1": 3.0986498089635717e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06533645982247885
      },
      {
        "k": 5,
        "preference": 19,
        "l1": 3.0460660035980247e-16,
        "max_abs": 1.4094628242311558e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12395949525394216
      },
      {
        "k": 5,
        "preference": 20,
        "l1": 3.1794228708137418e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.05331355180258966
      },
      {
        "k": 5,
        "preference": 21,
        "l1": 2.734357879008442e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14093045182546313
      },
      {
        "k": 5,
        "preference": 22,
        "l1": 4.2256779672622535e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.08248213203656998
      },
      {
        "k": 5,
        "preference": 23,
        "l1": 3.4726995584710707e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04812992855791293
      },
      {
        "k": 5,
        "preference": 24,
        "l1": 2.5869063835504136e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12464740751492326
      },
      {
        "k": 5,
        "preference": 25,
        "l1": 2.579316968343015e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0361033986386832
      },
      {
        "k": 5,
        "preference": 26,
        "l1": 2.990229591715021e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11812999905930727
      },
      {
        "k": 5,
        "preference": 27,
        "l1": 3.4862520856271395e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.0504882687024224
      },
      {
        "k": 5,
        "preference": 28,
        "l1": 4.0288952729561345e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13012162387562748
      },
      {
        "k": 5,
        "preference": 29,
        "l1": 2.459512628283367e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13472293650843686
      },
      {
        "k": 5,
        "preference": 30,
        "l1": 3.2016490153496946e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.17358448777878913
      },
      {
        "k": 5,
        "preference": 31,
        "l1": 2.8327492261615017e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1406613418750957
      },
      {
        "k": 5,
        "preference": 32,
        "l1": 3.3355479836516544e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.05234271388960136
      },
      {
        "k": 5,
        "preference": 33,
        "l1": 3.1452705023804484e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11872045183349424
      },
      {
        "k": 5,
        "preference": 34,
        "l1": 2.2957981002380556e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07431607840359308
      },
      {
        "k": 5,
        "preference": 35,
        "l1": 3.8445809036335987e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07600475409739459
      },
      {
        "k": 5,
        "preference": 36,
        "l1": 4.261998740040518e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09485212521682433
      },
      {
        "k": 5,
        "preference": 37,
        "l1": 3.294890402183448e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14273909658216274
      },
      {
        "k": 5,
        "preference": 38,
        "l1": 3.5626883387873676e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09322841959787671
      },
      {
        "k": 5,
        "preference": 39,
        "l1": 3.9519169187096637e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12640950101014764
      },
      {
        "k": 5,
        "preference": 40,
        "l1": 3.4342103813478353e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07257208566086715
      },
      {
        "k": 5,
        "preference": 41,
        "l1": 2.1488887058662698e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1387650647369726
      },
      {
        "k": 5,
        "preference": 42,
        "l1": 2.653584817158272e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.15651605263242363
      },
      {
        "k": 5,
        "preference": 43,
        "l1": 2.973424458041496e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.16299155954994926
      },
      {
        "k": 5,
        "preference": 44,
        "l1": 3.5063098258181213e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12227012518408012
      },
      {
        "k": 5,
        "preference": 45,
        "l1": 3.2189962501094627e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08947707737219252
      },
      {
        "k": 5,
        "preference": 46,
        "l1": 2.6869240339622014e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.14382870484509588
      },
      {
        "k": 5,
        "preference": 47,
        "l1": 2.7885679876327174e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11105880546566506
      },
      {
        "k": 5,
        "preference": 48,
        "l1": 2.397713104451693e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.025920608591904072
      },
      {
        "k": 5,
        "preference": 49,
        "l1": 4.3194614551822497e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04148645433149297
      },
      {
        "k": 5,
        "preference": 50,
        "l1": 3.465652244349915e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.133343724949685
      },
      {
        "k": 5,
        "preference": 51,
        "l1": 3.4488471106763896e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11663326997870159
      },
      {
        "k": 5,
        "preference": 52,
        "l1": 4.777536873057375e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08724818555447805
      },
      {
        "k": 5,
        "preference": 53,
        "l1": 3.3263322651855276e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03762489579703519
      },
      {
        "k": 5,
        "preference": 54,
        "l1": 4.456070928915423e-16,
        "max_abs": 1.4094628242311558e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12740774226593
      },
      {
        "k": 5,
        "preference": 55,
        "l1": 2.918672248330978e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09451159261698952
      },
      {
        "k": 5,
        "preference": 56,
        "l1": 6.353695781308177e-16,
        "max_abs": 3.469446951953614e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09398850333513174
      },
      {
        "k": 5,
        "preference": 57,
        "l1": 2.535406780357352e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.05296234494300135
      },
      {
        "k": 5,
        "preference": 58,
        "l1": 3.1517757154153614e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14579124113197492
      },
      {
        "k": 5,
        "preference": 59,
        "l1": 3.1138286393783687e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.15273783587677853
      },
      {
        "k": 5,
        "preference": 60,
        "l1": 4.994919408640719e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.02466724286314732
      },
      {
        "k": 5,
        "preference": 61,
        "l1": 4.825241768646737e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06318947996684796
      },
      {
        "k": 5,
        "preference": 62,
        "l1": 4.754226526348937e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07804409429852366
      },
      {
        "k": 5,
        "preference": 63,
        "l1": 3.191891195797325e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11598024610447827
      },
      {
        "k": 5,
        "preference": 64,
        "l1": 3.4423418976414766e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08388263094117913
      },
      {
        "k": 5,
        "preference": 65,
        "l1": 3.780612975456954e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12256838358606216
      },
      {
        "k": 5,
        "preference": 66,
        "l1": 3.402226417259513e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.0590871503773681
      },
      {
        "k": 5,
        "preference": 67,
        "l1": 3.7724814591633127e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10017172036094431
      },
      {
        "k": 5,
        "preference": 68,
        "l1": 2.9848085808525937e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04632847150429872
      },
      {
        "k": 5,
        "preference": 69,
        "l1": 2.9235511581071627e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1566176676493853
      },
      {
        "k": 5,
        "preference": 70,
        "l1": 2.4296970685400154e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.07292420601612262
      },
      {
        "k": 5,
        "preference": 71,
        "l1": 4.603522424373452e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10439626791463613
      },
      {
        "k": 5,
        "preference": 72,
        "l1": 3.2672432467850676e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14494908001368562
      },
      {
        "k": 5,
        "preference": 73,
        "l1": 2.240503789441295e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1498136026122117
      },
      {
        "k": 5,
        "preference": 74,
        "l1": 3.260195932663912e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07405002013640158
      },
      {
        "k": 5,
        "preference": 75,
        "l1": 2.277908764392045e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.053151346005228736
      },
      {
        "k": 5,
        "preference": 76,
        "l1": 4.1167156489274603e-16,
        "max_abs": 2.6020852139652106e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.10601327337330085
      },
      {
        "k": 5,
        "preference": 77,
        "l1": 2.7137580377312176e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.05412140634028056
      },
      {
        "k": 5,
        "preference": 78,
        "l1": 2.2177355438190993e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.037696328190212035
      },
      {
        "k": 5,
        "preference": 79,
        "l1": 3.0574501264091225e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.05036475271754565
      },
      {
        "k": 5,
        "preference": 80,
        "l1": 3.9155961459313993e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.053352916100525886
      },
      {
        "k": 5,
        "preference": 81,
        "l1": 2.3014901616436045e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12446466596713224
      },
      {
        "k": 5,
        "preference": 82,
        "l1": 4.864815147942458e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.031633519747506354
      },
      {
        "k": 5,
        "preference": 83,
        "l1": 3.9811903773667723e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11314072977575973
      },
      {
        "k": 5,
        "preference": 84,
        "l1": 5.152128723651117e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03422611831358209
      },
      {
        "k": 5,
        "preference": 85,
        "l1": 2.355971320811001e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09820256142323168
      },
      {
        "k": 5,
        "preference": 86,
        "l1": 3.329584871702984e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09069693760615757
      },
      {
        "k": 5,
        "preference": 87,
        "l1": 2.9826401765076227e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10916766543877574
      },
      {
        "k": 5,
        "preference": 88,
        "l1": 2.8210940528072825e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 8.526512829121202e-14,
        "without_reaching_l1": 0.06353247120980915
      },
      {
        "k": 5,
        "preference": 89,
        "l1": 3.765434145042157e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.061401879101781756
      },
      {
        "k": 5,
        "preference": 90,
        "l1": 3.209238430557093e-16,
        "max_abs": 2.3852447794681098e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1464798928729436
      },
      {
        "k": 5,
        "preference": 91,
        "l1": 3.3566899260151217e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14886482307382337
      },
      {
        "k": 5,
        "preference": 92,
        "l1": 2.8763883636040433e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.12608080163696117
      },
      {
        "k": 5,
        "preference": 93,
        "l1": 4.1009947174264205e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.042479134797483414
      },
      {
        "k": 5,
        "preference": 94,
        "l1": 2.5055912206140007e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13512939914751324
      },
      {
        "k": 5,
        "preference": 95,
        "l1": 3.114912841550854e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.13105536355903297
      },
      {
        "k": 5,
        "preference": 96,
        "l1": 3.754050022231059e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.043312854062774735
      },
      {
        "k": 5,
        "preference": 97,
        "l1": 4.195862407518902e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14901483987384678
      },
      {
        "k": 5,
        "preference": 98,
        "l1": 4.4246290659133436e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 2.2737367544323206e-13,
        "without_reaching_l1": 0.09067226478620194
      },
      {
        "k": 5,
        "preference": 99,
        "l1": 3.1669545458301585e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12295025024620958
      },
      {
        "k": 5,
        "preference": 100,
        "l1": 2.55275401511712e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.18731527419013932
      },
      {
        "k": 5,
        "preference": 101,
        "l1": 2.629732369363591e-16,
        "max_abs": 1.1926223897340549e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.10459200049163139
      },
      {
        "k": 5,
        "preference": 102,
        "l1": 4.469081354985249e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.15634639152347804
      },
      {
        "k": 5,
        "preference": 103,
        "l1": 2.7712207528729493e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07956271761799436
      },
      {
        "k": 5,
        "preference": 104,
        "l1": 2.4096393283490336e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1601231689226392
      },
      {
        "k": 5,
        "preference": 105,
        "l1": 2.6828582758153807e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.08113045908851096
      },
      {
        "k": 5,
        "preference": 106,
        "l1": 4.51461784622964e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 2.842170943040401e-14,
        "without_reaching_l1": 0.07952772340697112
      },
      {
        "k": 5,
        "preference": 107,
        "l1": 4.1470733097570545e-16,
        "max_abs": 1.734723475976807e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.05791319454404567
      },
      {
        "k": 5,
        "preference": 108,
        "l1": 3.4732416595573135e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.062282439849715716
      },
      {
        "k": 5,
        "preference": 109,
        "l1": 4.281514379145257e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.19463103280421612
      },
      {
        "k": 5,
        "preference": 110,
        "l1": 2.905119721174909e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.050780070900367714
      },
      {
        "k": 5,
        "preference": 111,
        "l1": 2.7180948464211596e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.14278389668567879
      },
      {
        "k": 5,
        "preference": 112,
        "l1": 2.940898392866931e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.02626103152419349
      },
      {
        "k": 5,
        "preference": 113,
        "l1": 8.317456966222547e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.03281399157563121
      },
      {
        "k": 5,
        "preference": 114,
        "l1": 3.857591329703425e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.06784361043215162
      },
      {
        "k": 5,
        "preference": 115,
        "l1": 3.767602549387128e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.07970680907006127
      },
      {
        "k": 5,
        "preference": 116,
        "l1": 4.001790218643997e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.09913483715376023
      },
      {
        "k": 5,
        "preference": 117,
        "l1": 3.318742849978129e-16,
        "max_abs": 1.5178830414797062e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.06022618860518616
      },
      {
        "k": 5,
        "preference": 118,
        "l1": 2.603169416137696e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.11404821013447257
      },
      {
        "k": 5,
        "preference": 119,
        "l1": 4.189899295570232e-16,
        "max_abs": 2.8189256484623115e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1603075887107374
      },
      {
        "k": 5,
        "preference": 120,
        "l1": 5.206880933361635e-16,
        "max_abs": 1.951563910473908e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.11247848455015921
      },
      {
        "k": 5,
        "preference": 121,
        "l1": 3.268327448957553e-16,
        "max_abs": 2.168404344971009e-18,
        "flow_certificate_residual": 8.526512829121202e-14,
        "without_reaching_l1": 0.09536948212879798
      },
      {
        "k": 5,
        "preference": 122,
        "l1": 2.425902360936316e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.16399676457706464
      },
      {
        "k": 5,
        "preference": 123,
        "l1": 2.076247160309741e-16,
        "max_abs": 8.673617379884035e-19,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.04856665516017433
      },
      {
        "k": 5,
        "preference": 124,
        "l1": 2.570643350963131e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.12530576684042985
      },
      {
        "k": 5,
        "preference": 125,
        "l1": 2.6546690193307576e-16,
        "max_abs": 1.0842021724855044e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.1831940541259599
      },
      {
        "k": 5,
        "preference": 126,
        "l1": 2.8601253310167607e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 5.684341886080802e-14,
        "without_reaching_l1": 0.09554705246754837
      },
      {
        "k": 5,
        "preference": 127,
        "l1": 3.057721176952244e-16,
        "max_abs": 1.3010426069826053e-18,
        "flow_certificate_residual": 1.1368683772161603e-13,
        "without_reaching_l1": 0.1486535332947301
      }
    ]
  },
  "claim_3": {
    "operators": [
      "linear scalarization",
      "harmonic mean",
      "contrast"
    ],
    "nonlinear": [
      {
        "operator": "harmonic",
        "ingredients": [
          "shubert",
          "sphere"
        ],
        "l1": 0.10200658914352616,
        "distortion_identity_max_abs": 4.336808689942018e-19,
        "without_reaching_l1": 0.11514693367803527,
        "favored_mass_induced": 0.4069665171871615,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0,
        "distribution_min": 6.624294491165516e-06
      },
      {
        "operator": "harmonic",
        "ingredients": [
          "shubert",
          "diagonal"
        ],
        "l1": 0.09778294904874857,
        "distortion_identity_max_abs": 4.336808689942018e-19,
        "without_reaching_l1": 0.16333151267866336,
        "favored_mass_induced": 0.45386883606293993,
        "favored_mass_uniform": 0.2509765625,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 3.766158629442965e-07
      },
      {
        "operator": "harmonic",
        "ingredients": [
          "branin",
          "sphere"
        ],
        "l1": 0.13023776039170296,
        "distortion_identity_max_abs": 4.336808689942018e-19,
        "without_reaching_l1": 0.13100768095306758,
        "favored_mass_induced": 0.3542903736675836,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 7.031132960593508e-09
      },
      {
        "operator": "harmonic",
        "ingredients": [
          "circle1",
          "circle2"
        ],
        "l1": 0.2796037312537852,
        "distortion_identity_max_abs": 1.734723475976807e-18,
        "without_reaching_l1": 0.4976907849488701,
        "favored_mass_induced": 0.9296217011743364,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 3.4025973703852106e-05
      },
      {
        "operator": "harmonic",
        "ingredients": [
          "circle1",
          "circle3"
        ],
        "l1": 0.18209147735655623,
        "distortion_identity_max_abs": 1.734723475976807e-18,
        "without_reaching_l1": 0.4444749823605463,
        "favored_mass_induced": 0.8856901694895574,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 0.9999999999999996,
        "distribution_min": 5.650442525732901e-05
      },
      {
        "operator": "harmonic",
        "ingredients": [
          "circle2",
          "circle3"
        ],
        "l1": 0.22709892707610368,
        "distortion_identity_max_abs": 1.734723475976807e-18,
        "without_reaching_l1": 0.457828712288448,
        "favored_mass_induced": 0.9117035530487688,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 3.911075271615389e-05
      },
      {
        "operator": "contrast",
        "ingredients": [
          "shubert",
          "sphere"
        ],
        "l1": 0.08660785983761297,
        "distortion_identity_max_abs": 4.336808689942018e-19,
        "without_reaching_l1": 0.11259089139833225,
        "favored_mass_induced": 0.44633652359742765,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 0.9999999999999998,
        "distribution_min": 1.6225254805595712e-06
      },
      {
        "operator": "contrast",
        "ingredients": [
          "shubert",
          "diagonal"
        ],
        "l1": 0.03890162824867699,
        "distortion_identity_max_abs": 4.336808689942018e-19,
        "without_reaching_l1": 0.394987890854585,
        "favored_mass_induced": 0.47335421897144425,
        "favored_mass_uniform": 0.2509765625,
        "distribution_sum": 0.9999999999999998,
        "distribution_min": 1.4471999100519369e-06
      },
      {
        "operator": "contrast",
        "ingredients": [
          "branin",
          "sphere"
        ],
        "l1": 0.09576825693889122,
        "distortion_identity_max_abs": 4.336808689942018e-19,
        "without_reaching_l1": 0.11647433133626657,
        "favored_mass_induced": 0.40669068369469374,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0,
        "distribution_min": 8.925648291881825e-15
      },
      {
        "operator": "contrast",
        "ingredients": [
          "circle1",
          "circle2"
        ],
        "l1": 0.28803862328322205,
        "distortion_identity_max_abs": 8.673617379884035e-19,
        "without_reaching_l1": 0.26180720503853383,
        "favored_mass_induced": 0.9816960477110961,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 1.2020411769453254e-09
      },
      {
        "operator": "contrast",
        "ingredients": [
          "circle1",
          "circle3"
        ],
        "l1": 0.15299393580626713,
        "distortion_identity_max_abs": 8.673617379884035e-19,
        "without_reaching_l1": 0.4688738009488177,
        "favored_mass_induced": 0.98339579500814,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 3.017707198683282e-09
      },
      {
        "operator": "contrast",
        "ingredients": [
          "circle2",
          "circle3"
        ],
        "l1": 0.1839188927984333,
        "distortion_identity_max_abs": 8.673617379884035e-19,
        "without_reaching_l1": 0.43752843328110147,
        "favored_mass_induced": 0.9693595873010068,
        "favored_mass_uniform": 0.25,
        "distribution_sum": 1.0000000000000002,
        "distribution_min": 2.141469498636017e-09
      }
    ],
    "all_favored_enriched_over_uniform": true
  },
  "negative_controls": {
    "omit_reaching_l1": 0.15662801795415238,
    "omit_partition_l1": 0.009328807638861777,
    "apply_beta2_to_beta1_target_l1": 0.07017473043675532
  }
}

````


````output
{
  "scale": {
    "grid": "32x32",
    "states": 1024,
    "ingredients": 8,
    "linear_settings": 512,
    "nonlinear_settings": 12
  },
  "wall_seconds": 5.186631512129679,
  "linear_max_l1": 8.317456966222547e-16,
  "linear_max_abs": 4.336808689942018e-18,
  "ablation_median_l1": 0.11472514991085364,
  "nonlinear_l1_range": [
    0.03890162824867699,
    0.28803862328322205
  ],
  "controls": {
    "omit_reaching_l1": 0.15662801795415238,
    "omit_partition_l1": 0.009328807638861777,
    "apply_beta2_to_beta1_target_l1": 0.07017473043675532
  }
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5ed91180a227", "created_at": "2026-07-17T04:04:00+00:00", "title": "Across 128 preferences at each of k=2,3,4,5 objectives (512 total), maximum L1…"}
-->
Across 128 preferences at each of k=2,3,4,5 objectives (512 total), maximum L1 error is 8.32e-16 and maximum pointwise error is 4.34e-18. A separate incoming/outgoing/terminal edge-flow certificate checks every state.
