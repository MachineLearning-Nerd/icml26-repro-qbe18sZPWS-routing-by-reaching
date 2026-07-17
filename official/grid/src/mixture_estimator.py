"""Mixture of GFlowNets for multi-objective optimization.

Combines multiple pre-trained GFlowNets at inference time to approximate
different target distributions without retraining.

Mixing Types (with flow weighting u(s)):
    - scalarization: (Σ w_i (Z_i u_i p_iF)^{1/β})^β — beta-correct
    - harmonic_mean: Flow-weighted HM of u_i · p_iF
    - contrast: Flow-weighted contrast of u_i · p_iF

Without flow weighting (policy-only, *_without_u):
    - scalarization_without_u: Σ w_i · Z_i · p_iF
    - harmonic_mean_without_u: HM of p_iF
    - contrast_without_u: Contrast of p_iF

flow_estimation (for with-u variants only):
    - "logF"           : read u, F directly from gfn.logF (model_f)
    - "detailed_balance": compute log F(s) via detailed balance DP (otf)
"""

import torch
import torch.nn as nn

from gfn.states import DiscreteStates, States
from gfn.utils.distributions import UnsqueezedCategorical

from .utils import compute_log_F_by_detailed_balance


class MixtureDiscretePolicyEstimator(nn.Module):
    """Mixture policy estimator combining multiple pre-trained GFlowNets.

    Precomputes F(s), u(s)=F(s)/Z for all states at init time, then mixes
    policies at each step.

    Args:
        gfn_list: list of pre-trained SubTBGFlowNet models
        n_actions: env.n_actions
        preprocessor: state preprocessor
        loss_type: kept for back-compat (currently only "SubTB" supported)
        is_backward: False for forward policy mixing
        mixing_type: see module docstring
        env: the environment (for state grid / flows)
        weight_list: mixing weights (length = len(gfn_list))
        beta: reward sharpening exponent used during base GFN training
        flow_estimation: "logF" (default) or "detailed_balance"
    """

    def __init__(self, gfn_list, n_actions, preprocessor, loss_type=None,
                 is_backward=False, mixing_type=None, env=None,
                 weight_list=None, beta=1.0, flow_estimation="logF"):
        nn.Module.__init__(self)
        self.preprocessor = preprocessor
        self.is_backward = is_backward
        self.n_actions = n_actions
        self.weights = torch.tensor(weight_list, device=env.device)
        self.gfn_list = gfn_list
        self.mixing_type = mixing_type
        self.beta = beta
        self.flow_estimation = flow_estimation

        self.fs, self.us, self.zs = self._compute_flows(gfn_list, env)

    def _compute_flows(self, gfn_list, env):
        """Precompute F(s) and u(s) = F(s) / Z for every state, each model."""
        H = env.height
        s0_state = env.states_from_tensor(env.s0)

        fs_list, zs_list = [], []
        for gfn in gfn_list:
            z = gfn.logF(s0_state).exp().item()
            if self.flow_estimation == "logF":
                f = gfn.logF(env.all_states).exp().squeeze()
            elif self.flow_estimation == "detailed_balance":
                f = compute_log_F_by_detailed_balance(env, gfn).exp().to(env.device)
            else:
                raise ValueError(f"Unknown flow_estimation: {self.flow_estimation}")
            fs_list.append(f.reshape(H, H).to(env.device))
            zs_list.append(z)

        zs = torch.tensor(zs_list, device=env.device)
        fs = torch.stack(fs_list)
        us = fs / zs.view(-1, 1, 1)
        return fs, us, zs

    def forward(self, input: States) -> torch.Tensor:
        """Per-model action logits. Shape: [n_models, n_states, n_actions]."""
        preprocessed = self.preprocessor(input)
        outs = [gfn.pf.module(preprocessed) for gfn in self.gfn_list]
        return torch.stack(outs, dim=0)

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: torch.Tensor,
        sf_bias: float = 0.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> UnsqueezedCategorical:
        masks = states.backward_masks if self.is_backward else states.forward_masks
        logits = module_output.clone()
        logits[~masks.unsqueeze(0).expand_as(logits)] = -float("inf")

        probs_per_model = torch.softmax(logits, dim=-1)

        if self.mixing_type == "scalarization":
            probs = self._scalarization(states, logits)
        elif self.mixing_type == "harmonic_mean":
            probs = self._harmonic_mean(probs_per_model, states, use_flow=True)
        elif self.mixing_type == "contrast":
            probs = self._contrast(probs_per_model, states, use_flow=True)
        elif self.mixing_type == "scalarization_without_u":
            probs = self._scalarization_without_u(probs_per_model)
        elif self.mixing_type == "harmonic_mean_without_u":
            probs = self._harmonic_mean(probs_per_model, states, use_flow=False)
        elif self.mixing_type == "contrast_without_u":
            probs = self._contrast(probs_per_model, states, use_flow=False)
        else:
            raise ValueError(f"Unknown mixing_type: {self.mixing_type}")

        return UnsqueezedCategorical(probs=probs)

    # === Mixing strategies ===

    def _get_weighted_pf(self, probs_per_model, states, use_flow):
        """Optionally multiply per-model probs by u(s).

        u_i(s) · p_iF(a|s)  if use_flow else  p_iF(a|s)
        (Z_i cancels in HM/contrast ratios, so we use u, not F.)
        """
        if use_flow:
            x, y = states.tensor[:, 0].long(), states.tensor[:, 1].long()
            us = self.us[:, x, y]
            return us.unsqueeze(-1) * probs_per_model
        return probs_per_model

    def _harmonic_mean(self, probs_per_model, states, use_flow):
        """Harmonic mean: prod(x_i) / sum(prod_{j≠i}(x_j))."""
        u_pf = self._get_weighted_pf(probs_per_model, states, use_flow)
        n = len(self.gfn_list)
        numerator = u_pf.prod(dim=0)
        denominator = sum(numerator / (u_pf[j] + 1e-10) for j in range(n))
        return numerator / (denominator + 1e-10)

    def _contrast(self, probs_per_model, states, use_flow):
        """Contrast: sharpen towards the first model."""
        u_pf = self._get_weighted_pf(probs_per_model, states, use_flow)
        n = len(self.gfn_list)
        if n == 2:
            return u_pf[0] ** 2 / (u_pf.sum(dim=0) + 1e-10)
        elif n == 3:
            p1, p2, p3 = u_pf[0], u_pf[1], u_pf[2]
            return p1 ** 4 / ((p1 + p2) * (p1 ** 2 + p1 * p3 + p2 * p3) + 1e-10)
        raise ValueError(f"Contrast only supports 2 or 3 models, got {n}")

    def _scalarization_without_u(self, probs_per_model):
        """Σ w_i · Z_i · p_iF(a|s), renormalized."""
        w = self.weights.view(-1, 1, 1)
        zs = self.zs.view(-1, 1, 1)
        probs = (w * zs * probs_per_model).sum(dim=0)
        return probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    def _scalarization(self, states, logits):
        """(Σ w_i (F_i p_iF)^{1/β})^β in log space. Beta-correct."""
        n_models = len(self.gfn_list)
        x, y = states.tensor[:, 0].long(), states.tensor[:, 1].long()
        fs = self.fs[:, x, y]

        log_pf = torch.log_softmax(logits, dim=-1)
        log_fs = torch.log(fs + 1e-30)
        log_w = torch.log(self.weights + 1e-30)

        log_terms = log_w.view(n_models, 1, 1) + (log_pf + log_fs.unsqueeze(-1)) / self.beta
        log_probs = self.beta * torch.logsumexp(log_terms, dim=0)
        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
        return torch.exp(log_probs)

    @property
    def expected_output_dim(self) -> int:
        return self.n_actions - 1 if self.is_backward else self.n_actions
