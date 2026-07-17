"""
Mixture sampling for TB-trained GFlowNets with detailed-balance state-flow estimation.

TB models don't learn F(s) explicitly. Instead, we estimate F(s) along the
generation trajectory using the detailed balance relation:

    F(s') = F(s) * P_F(s'|s) / P_B(s|s')

In log space with uniform P_B (P_B = 1/n_back):

    log F(s_{t+1}) = log F(s_t) + log P_F(s_{t+1}|s_t) + log(n_back(s_{t+1}))

Starting from log F(s_0) = log Z (obtained from model.logZ()).
"""
import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gflownet.envs.graph_building_env import GraphAction, GraphActionType
from gflownet.utils.misc import get_worker_device


def get_log_z_tensor(trainers, ctx, env):
    """Get log Z from TB models via model.logZ()."""
    dev = get_worker_device()
    log_z_list = []
    for t in trainers:
        ci = t.task.sample_conditional_information(1, 0)
        log_z = t.model.logZ(ci["encoding"].to(dev))  # [1, 1]
        log_z_list.append(log_z.squeeze())  # scalar
    return torch.stack(log_z_list).unsqueeze(0)  # [1, n_models]


class MixtureGraphSamplerTB:
    """Mixture sampler for TB-trained GFlowNets — scalarization only.

    TB models do not learn a per-state F(s) head, so we estimate F(s) on the fly
    along the generation trajectory via the detailed-balance relation
    `F(s') = F(s) * P_F(s'|s) / P_B(s|s')` (see module docstring). HM and contrast
    modes need F(s) too but are not exposed here; if you need them, train SubTB
    ingredients and use `MixtureGraphSampler` from `mixture_graph_sampling_subtb.py`.
    """

    def __init__(
        self,
        ctx,
        env,
        max_len,
        max_nodes,
        sample_temp=1,
        correct_idempotent=False,
        pad_with_terminal_state=False,
        ingredient_model_trainers=None,
        beta=32,
        without_u=False,
    ):
        """
        Parameters
        ----------
        ctx: GraphBuildingEnvContext
        env: GraphBuildingEnv
        max_len: int
            Max trajectory length (`None` falls back to 128).
        max_nodes: int
            Max graph size (`None` falls back to 128); exceeding it kills the trajectory.
        sample_temp: float
            [Experimental] Softmax temperature used when sampling.
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting.
        pad_with_terminal_state: bool
            [Experimental] If True, pad trajectories with a terminal Stop.
        ingredient_model_trainers: List[Trainer]
            Pre-trained TB single-objective GFN trainers, one per objective. Order must
            match the preferences passed via `total_cond_info["preferences"]` at sample time.
        beta: float
            Temperature for scalarization (higher β → sharper max-like combination).
        without_u: bool
            If True, drop log u from the per-ingredient term.
            Final form becomes `logsumexp_i(w_i + log Z_i + pf_i)`.
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state
        self.beta = beta
        self.without_u = without_u

        dev = get_worker_device()
        self.ingredient_model_trainers = ingredient_model_trainers
        for t in self.ingredient_model_trainers:
            t.model.to(dev)
        self.log_z_tensor = get_log_z_tensor(
            self.ingredient_model_trainers, self.ctx, self.env
        )  # [1, n_models]

    def sample_from_model(
        self,
        model: nn.Module,
        n: int,
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
        total_cond_info=None,
    ):
        """Sample from a mixture of TB-trained GFlowNets via scalarization.

        Parameters
        ----------
        model: nn.Module
            Unused (kept for interface compatibility). Mixing uses ingredient models.
        n: int
            Number of graphs to sample.
        cond_info: Tensor
            Conditional information (unused, kept for compatibility).
        random_action_prob: float
            Probability of random action (unused).
        total_cond_info: dict
            Must contain 'preferences' key with shape [n, n_models].

        Returns
        -------
        data: List[Dict]
            List of trajectory dicts with keys: traj, fwd_logprob, bck_logprob, is_valid, etc.
        """
        dev = get_worker_device()
        n_models = len(self.ingredient_model_trainers)

        data = [
            {"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []}
            for _ in range(n)
        ]
        fwd_logprob: List[List[Tensor]] = [[] for _ in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for _ in range(n)]

        graphs = [self.env.new() for _ in range(n)]
        done = [False] * n
        bck_a = [[GraphAction(GraphActionType.Stop)] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        # ---- On-the-fly F estimation ----
        # Initialize: log F_i(s_0) = log Z_i for every sample and every model
        log_F_running = self.log_z_tensor.expand(n, -1).clone()  # [n, n_models]

        log_w_tensor = total_cond_info["preferences"].log().to(dev)  # [n, n_models]

        for t in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_graph_idx = [i for i in range(n) if not done[i]]
            torch_graphs_batch = self.ctx.collate(torch_graphs).to(dev)

            # Forward pass for each ingredient model
            fwd_cats = []
            for trainer in self.ingredient_model_trainers:
                ci = trainer.task.sample_conditional_information(
                    len(not_done_graph_idx), t
                )
                result = trainer.model(
                    torch_graphs_batch, ci["encoding"].to(dev)
                )
                f = result[0]  # fwd_cat (TB may return 2 or 3 values)
                f.logits = f.logsoftmax()
                fwd_cats.append(f)

            sample_idx_list = fwd_cats[0].batch

            # Current log F for active samples
            log_F_active = log_F_running[not_done_graph_idx]  # [n_active, n_models]

            # Build guided (mixed) categorical
            guided_cat = copy.copy(fwd_cats[0])
            guided_cat.logits = [
                torch.full_like(L, float("-inf")) for L in fwd_cats[0].logits
            ]

            tmp = []
            logits_all = [fwd.logits for fwd in fwd_cats]

            for fwds_and_idx in zip(*logits_all, sample_idx_list):
                fwds = fwds_and_idx[:-1]
                idx = fwds_and_idx[-1]

                if fwds[0].shape[0] == 0:
                    tmp.append(
                        torch.empty(
                            fwds[0].shape, dtype=torch.float32, device=dev
                        )
                    )
                    continue

                global_i = [not_done_graph_idx[local_i] for local_i in idx]
                ws = [log_w_tensor[global_i, i] for i in range(n_models)]

                # Scalarization:
                #   without_u=False: F_mix ∝ (Σ_i w_i · (pf_i · Z_i · u_i)^{1/β})^β
                #   without_u=True:  F_mix ∝ Σ_i w_i · pf_i · Z_i
                beta = 1.0 if self.without_u else self.beta
                log_Zs = [self.log_z_tensor[0, i] for i in range(n_models)]  # scalars
                terms = []
                for i, (pf, w) in enumerate(zip(fwds, ws)):
                    term = pf + log_Zs[i]
                    if not self.without_u:
                        # log u_i(s) = log F_i(s) - log Z_i, per active sample (DB-estimated F).
                        log_u_i = log_F_active[idx, i] - log_Zs[i]
                        term = term + log_u_i[:, None]
                    terms.append(w[:, None] + term / beta)
                final = beta * torch.logsumexp(torch.stack(terms, dim=0), dim=0)

                tmp.append(final)

            for i, logit in enumerate(tmp):
                guided_cat.logits[i] = logit

            # Sample action from the mixed distribution
            actions = guided_cat.sample()
            graph_actions = [
                self.ctx.ActionIndex_to_GraphAction(g, a)
                for g, a in zip(torch_graphs, actions)
            ]
            log_probs = guided_cat.log_prob(actions)

            # Per-model forward log probs of the taken actions (for F update)
            per_model_log_probs = [
                fwd_cats[m].log_prob(actions) for m in range(n_models)
            ]  # each: [n_active]

            # Step each trajectory
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))

                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                else:
                    gp = graphs[i]
                    try:
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]["is_valid"] = False
                        bck_logprob[i].append(
                            torch.tensor([1.0], device=dev).log()
                        )
                        data[i]["is_sink"].append(1)
                        continue

                    if t == self.max_len - 1:
                        done[i] = True

                    n_back = self.env.count_backward_transitions(
                        gp, check_idempotent=self.correct_idempotent
                    )
                    bck_logprob[i].append(
                        torch.tensor([1 / n_back], device=dev).log()
                    )
                    data[i]["is_sink"].append(0)

                    # ---- Update log F(s_{t+1}) for each model ----
                    # log F(s') = log F(s) + log P_F(s'|s) - log P_B(s|s')
                    # Uniform P_B: log P_B(s|s') = -log(n_back(s'))
                    # => log F(s') = log F(s) + log P_F(s'|s) + log(n_back(s'))
                    log_n_back = math.log(n_back)
                    for m in range(n_models):
                        log_F_running[i, m] += (
                            per_model_log_probs[m][j].item() + log_n_back
                        )

                    graphs[i] = gp

                if done[i] and self.sanitize_samples and not self.ctx.is_sane(
                    graphs[i]
                ):
                    data[i]["is_valid"] = False

            if all(done):
                break

        for i in range(n):
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                data[i]["traj"].append(
                    (graphs[i], GraphAction(GraphActionType.Stop))
                )
                data[i]["is_sink"].append(1)
        return data


def make_mixture_sampler_tb(algo, ingredient_model_trainers, *, beta=32, without_u=False):
    """Build a `MixtureGraphSamplerTB` from an existing TB-style `algo`.

    Use this at eval time to swap `algo.graph_sampler`:

        algo.graph_sampler = make_mixture_sampler_tb(algo, ingredient_trainers, beta=...)
    """
    return MixtureGraphSamplerTB(
        algo.ctx,
        algo.env,
        algo.global_cfg.algo.max_len,
        algo.global_cfg.algo.max_nodes,
        algo.sample_temp,
        correct_idempotent=algo.cfg.do_correct_idempotent,
        pad_with_terminal_state=algo.cfg.do_parameterize_p_b,
        ingredient_model_trainers=ingredient_model_trainers,
        beta=beta,
        without_u=without_u,
    )
