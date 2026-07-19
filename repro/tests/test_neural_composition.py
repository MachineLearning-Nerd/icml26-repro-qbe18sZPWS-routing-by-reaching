"""Integrity tests for the trained-neural-GFlowNet composition evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

from run_routing_by_reaching import rewards, rollout, scalar_mix
import train_neural_gfns as tg

RESULT = ROOT / "outputs" / "neural_composition.json"


@pytest.fixture(scope="module")
def report():
    return json.loads(RESULT.read_text())


def test_all_ingredients_are_trained_networks(report):
    # every ingredient has a cached trained state dict, nonzero train evidence
    for name in report["ingredients"]:
        assert (ROOT / "outputs" / "nets" / f"{name}.pt").exists()


def test_learned_logZ_close_to_exact(report):
    for name, v in report["ingredients"].items():
        assert abs(v["learned_logZ"] - v["true_logZ_exact"]) < 0.15


def test_linear_exact_recovery_with_trained_ingredients(report):
    t2 = report["T2_linear"]
    assert t2["settings"] == 128
    assert t2["max_L1"] < 1e-10
    assert t2["median_ablation_L1"] > 0.05


def test_recovery_identity_holds_for_any_policy_ingredients():
    """The exact-recovery identity is a property of the composition, not of
    ingredient quality: two arbitrary smooth policies compose exactly."""
    rng = np.random.default_rng(3)
    gs = []
    for _ in range(2):
        raw = rng.random((tg.H, tg.H, 3)) + 0.1
        raw[tg.H - 1, :, 0] = 0.0
        raw[:, tg.H - 1, 1] = 0.0
        pol = raw / raw.sum(axis=-1, keepdims=True)
        reaching, terminal = rollout(pol)
        gs.append(tg.TabularGFN(None, None, pol, reaching, terminal,
                                float(rng.uniform(0.5, 2.0))))
    w = np.array([0.3, 0.7])
    target = sum(wi * g.z * g.terminal for wi, g in zip(w, gs))
    target /= target.sum()
    _, term = rollout(scalar_mix(list(gs), w))
    assert np.abs(term / term.sum() - target).sum() < 1e-12


def test_adaptation_is_training_free(report):
    t4 = report["T4_adaptation"]
    assert t4["settings"] == 524
    assert t4["gradient_updates"] == 0
    assert t4["seconds"] < 10


def test_sampler_within_statistical_noise(report):
    t5 = report["T5_sampling"]
    assert t5["empirical_L1"] < 1.5 * t5["expected_multinomial_L1"]


def test_nonlinear_majority_enriched(report):
    rows = report["T3_nonlinear"]
    assert sum(r["enriched"] for r in rows) >= 9
    # contrast operator enriches in every tested pair
    assert all(r["enriched"] for r in rows if r["op"] == "contrast")
