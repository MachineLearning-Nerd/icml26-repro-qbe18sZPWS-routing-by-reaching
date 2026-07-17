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
