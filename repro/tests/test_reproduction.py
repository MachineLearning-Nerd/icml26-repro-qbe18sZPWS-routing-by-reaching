import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
from run_routing_by_reaching import exact_gfn, main, rewards, rollout, scalar_mix


@pytest.fixture(scope="session")
def result(tmp_path_factory):
    return main(tmp_path_factory.mktemp("run") / "summary.json")


def test_all_eight_official_rewards_present():
    assert len(rewards()) == 8


def test_exact_ingredient_distribution():
    for reward in rewards().values():
        g = exact_gfn(np.maximum(reward, 1e-12))
        assert np.abs(g.terminal - g.reward / g.reward.sum()).sum() < 1e-12


def test_policy_normalization():
    for reward in rewards().values():
        assert np.max(np.abs(exact_gfn(np.maximum(reward, 1e-12)).policy.sum(-1) - 1)) < 1e-14


def test_reaching_equals_flow_over_z():
    for reward in rewards().values():
        g = exact_gfn(np.maximum(reward, 1e-12))
        assert np.max(np.abs(g.reaching - g.flow / g.z)) < 1e-14


def test_rollout_mass_conservation():
    g = exact_gfn(np.maximum(rewards()["shubert"], 1e-12))
    _, terminal = rollout(g.policy)
    assert terminal.sum() == pytest.approx(1.0, abs=1e-14)


def test_claim1_no_retraining(result):
    assert result["claim_1"]["training_runs"] == 0
    assert result["claim_1"]["adaptation_settings"] == 524


def test_claim1_all_operator_distributions_valid(result):
    assert result["claim_1"]["all_distributions_valid"]


def test_claim2_all_512_full_grid_settings_exact(result):
    assert len(result["claim_2"]["settings"]) == 512
    assert result["claim_2"]["max_l1"] < 1e-12


def test_claim2_pointwise_exact(result):
    assert result["claim_2"]["max_abs"] < 1e-14


def test_claim2_independent_flow_certificate(result):
    assert result["claim_2"]["max_flow_certificate_residual"] < 1e-11


def test_claim2_covers_two_to_five_objectives(result):
    assert {row["k"] for row in result["claim_2"]["settings"]} == {2, 3, 4, 5}


def test_claim3_all_operator_families(result):
    assert result["claim_3"]["operators"] == ["linear scalarization", "harmonic mean", "contrast"]


def test_claim3_twelve_nonlinear_full_grid_settings(result):
    assert len(result["claim_3"]["nonlinear"]) == 12
    assert result["claim_3"]["all_favored_enriched_over_uniform"]


def test_claim3_distortion_identity(result):
    assert max(row["distortion_identity_max_abs"] for row in result["claim_3"]["nonlinear"]) < 1e-14


def test_negative_control_omit_reaching_rejected(result):
    assert result["negative_controls"]["omit_reaching_l1"] > 1e-2


def test_negative_control_omit_partition_rejected(result):
    assert result["negative_controls"]["omit_partition_l1"] > 1e-3


def test_negative_control_wrong_beta_rejected(result):
    assert result["negative_controls"]["apply_beta2_to_beta1_target_l1"] > 1e-3
