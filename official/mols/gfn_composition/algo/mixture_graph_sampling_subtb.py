import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gflownet.envs.graph_building_env import GraphAction, GraphActionType
from gflownet.utils.misc import get_worker_device, get_worker_rng


def get_log_z_tensor(trainers, ctx, env):
    dev = get_worker_device()
    log_z_list = []
    for t in trainers:
        ci = t.task.sample_conditional_information(1, 0)
        s0 = [ctx.graph_to_Data(env.new())]
        _, *_, log_z_pred = t.model(ctx.collate(s0).to(dev), ci['encoding'].to(dev))
        log_z_list.append(log_z_pred)
    log_z_tensor = torch.stack(log_z_list, dim=-1).squeeze(0) 
    return log_z_tensor # [1, n_models]

class MixtureGraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""

    def __init__(
        self, ctx, env, max_len, max_nodes, sample_temp=1,
        correct_idempotent=False, pad_with_terminal_state=False, ingredient_model_trainers=None,
        mode="scalarization", beta=32, dominant_model_idx=0, without_u=False,
        use_db_f=False,
    ):
        """
        Parameters
        ----------
        ctx: GraphBuildingEnvContext
            A context.
        env: GraphBuildingEnv
            A graph environment.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        sample_temp: float
            [Experimental] Softmax temperature used when sampling.
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting.
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal.
        ingredient_model_trainers: List[Trainer]
            Pre-trained single-objective GFN trainers, one per objective. The order must
            match the preferences passed via `total_cond_info["preferences"]` at sample time.
        mode: {"scalarization", "harmonic_mean", "contrast"}
            How to combine ingredient flows at each step:
              - "scalarization":   F_mix ∝ (Σ_i w_i F_i^{1/β})^β     (preference-weighted)
              - "harmonic_mean":   harmonic mean of u_i · P_F,i       (pref-agnostic AND-like)
              - "contrast":        p1-dominant ratio                   (pref-agnostic, asymmetric)
        beta: float
            Temperature for the scalarization mode. Higher β → sharper (max-like) combination.
        dominant_model_idx: int
            (contrast mode only) Which ingredient acts as p1 in the contrast formula.
        without_u: bool
            If True, drop log u from the per-ingredient term. For scalarization the
            final form becomes `logsumexp_i(w_i + log Z_i + pf_i)`.
            For HM and contrast modes, this drops the u factor from each ingredient term.
        use_db_f: bool
            If True, estimate F(s) online via F(s_{t+1}) = F(s_t) · P_F · n_back rather
            than using the model's saved per-graph F head.
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state
        self.mode = mode
        self.beta = beta
        self.dominant_model_idx = dominant_model_idx  # For contrast mode: which model is p1 (dominant)
        self.without_u = without_u
        self.use_db_f = use_db_f

        # For mixing
        dev = get_worker_device()
        self.ingredient_model_trainers = ingredient_model_trainers
        for t in self.ingredient_model_trainers:
            t.model.to(dev)
        self.log_z_tensor = get_log_z_tensor(
            self.ingredient_model_trainers,
            self.ctx,
            self.env,
        )   # [1, n_models]


    def sample_from_model(self, model: nn.Module, n: int, cond_info: Optional[Tensor],
                          random_action_prob: float = 0.0, total_cond_info=None):
        """Sample n trajectories by running each step's action through the mixture rule.

        The `model` argument is used only for its env/ctx shell — the action logits at each
        step are computed from the ingredient_model_trainers and combined according to
        `self.mode`. `random_action_prob` is accepted for upstream-API compatibility but is
        not currently applied; mixing eval uses the mixture distribution directly.

        Parameters
        ----------
        model: nn.Module
            Placeholder forward model (its policy is overridden by the mixture). Only used
            so the upstream `create_training_data_from_own_samples` API works.
        n: int
            Number of trajectories to sample.
        cond_info: Tensor
            Encoded conditioning, shape (n, num_cond_dim). Unused inside this method but
            kept for upstream-API compatibility.
        random_action_prob: float
            Unused (kept for upstream-API compatibility).
        total_cond_info: Dict[str, Tensor]
            Full cond_info dict from the task. MUST contain `"preferences"` of shape
            (n, n_ingredients) — these are the raw scalarization weights ω used in
            `mode="scalarization"`. In HM/CT modes the preferences are read but not used
            in the action-logit computation (a uniform 1/k bucket is fine).

        Returns
        -------
        data: List[Dict]
           One trajectory dict per sample, with keys: `traj`, `fwd_logprob`,
           `bck_logprob`, `bck_logprobs`, `is_sink`, `is_valid`, `result`, `bck_a`.
        """
        dev = get_worker_device()
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for _ in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for _ in range(n)]

        graphs = [self.env.new() for _ in range(n)]
        done = [False for _ in range(n)]
        # TODO: instead of padding with Stop, we could have a virtual action whose probability
        # always evaluates to 1. Presently, Stop should convert to a (0,0,0) ActionIndex, which should
        # always be at least a valid index, and will be masked out anyways -- but this isn't ideal.
        # Here we have to pad the backward actions with something, since the backward actions are
        # evaluated at s_{t+1} not s_t.
        bck_a = [[GraphAction(GraphActionType.Stop)] for _ in range(n)]
        rng = get_worker_rng()

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        """
            Mixture of ingredient models
        """
        n_models = len(self.ingredient_model_trainers)
        log_w_tensor = total_cond_info['preferences'].log().to(dev) # [n_batch, n_models]

        # Detailed-balance F: initialize log F(s_0) = log Z for each model
        if self.use_db_f:
            log_F_running = self.log_z_tensor.expand(n, -1).clone()  # [n, n_models]

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)] # graphs: list of length n_batch / torch_graphs: list of length K
            # before_data_graphs = [i for i in not_done(graphs)]
            not_done_graph_idx = [i for i in range(n) if not done[i]]     # List    
            # Forward pass to get GraphActionCategorical
            # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
            torch_graphs_batch = self.ctx.collate(torch_graphs).to(dev)

            fwd_cats = []
            log_reward_preds = []
            for trainer in self.ingredient_model_trainers:
                ci_beta_pref = trainer.task.sample_conditional_information(len(not_done_graph_idx), t)
                f, l = trainer.model(torch_graphs_batch, ci_beta_pref['encoding'].to(dev))
                f.logits = f.logsoftmax()
                fwd_cats.append(f)
                log_reward_preds.append(l)

            sample_idx_list = fwd_cats[0].batch

            # Copy a container to write guided logits
            guided_cat = copy.copy(fwd_cats[0])
            guided_cat.logits = [torch.full_like(L, float("-inf")) for L in fwd_cats[0].logits]

            tmp = []
            logits_all = [fwd.logits for fwd in fwd_cats]   # list of per-expert logits: len = n_experts
            for fwds_and_idx in zip(*logits_all, sample_idx_list):
                fwds = fwds_and_idx[:-1]
                idx = fwds_and_idx[-1]
                if fwds[0].shape[0] == 0:
                    empty = torch.empty(fwds[0].shape, dtype=torch.float32, device=fwds[0].device)
                    tmp.append(empty)
                    continue
                
                global_i = [not_done_graph_idx[local_i] for local_i in idx]

                # step1 : log F(s)
                if self.use_db_f:
                    log_F_tensor = log_F_running[not_done_graph_idx]  # [n_active, n_models]
                else:
                    log_F_tensor = torch.cat(log_reward_preds, dim=1) # [n_active, n_models]

                # step2 : operator
                if self.mode == "harmonic_mean":
                    if self.without_u:
                        log_terms = torch.stack(fwds, dim=0)  # [n_models, n_batch, k]
                    else:
                        log_Z = self.log_z_tensor.expand(log_F_tensor.shape[0], log_F_tensor.shape[1]) # [1, n_models] -> [n, n_models]
                        log_u_tensor = log_F_tensor - log_Z
                        log_us = [log_u_tensor[idx,i] for i in range(len(self.ingredient_model_trainers))]
                        log_terms = torch.stack([pf + u[:,None] for pf, u in zip(fwds, log_us)], dim=0)  # [n_models, n_batch, k]

                    # log(u_i * p_iF) for each model i: shape [n_models, n_batch, k]
                    # Numerator (log space): sum of all log(u_i * p_iF) = log(prod_i u_i * p_iF)
                    log_numerator = log_terms.sum(dim=0)  # [n_batch, k]
                    # Denominator (log space): logsumexp over (sum excluding j-th term) for each j
                    # For n models: HM = (prod_i x_i * n) / (sum_j prod_{i≠j} x_i)
                    # log(sum_j prod_{i≠j} x_i) = logsumexp_j(log_numerator - log_terms[j])
                    log_partial_sums = torch.stack([log_numerator - log_terms[j] for j in range(n_models)], dim=0)  # [n_models, n_batch, k]
                    log_denominator = torch.logsumexp(log_partial_sums, dim=0)  # [n_batch, k]
                    final = log_numerator - log_denominator  # [n_batch, k]

                elif self.mode == "contrast":
                    if self.without_u:
                        log_u_pf = torch.stack(fwds, dim=0)  # [n_models, n_batch, k]
                    else:
                        log_Z = self.log_z_tensor.expand(log_F_tensor.shape[0], log_F_tensor.shape[1]) # [1, n_models] -> [n, n_models]
                        log_u_tensor = log_F_tensor - log_Z
                        log_us = [log_u_tensor[idx,i] for i in range(len(self.ingredient_model_trainers))]
                        log_u_pf = torch.stack([pf + u[:,None] for pf, u in zip(fwds, log_us)], dim=0)  # [n_models, n_batch, k]

                    if n_models == 2:
                        # p1^2 / (p1 + p2)
                        log_numerator = 2 * log_u_pf[0]  # [n_batch, k]
                        log_denominator = torch.logsumexp(log_u_pf, dim=0)  # [n_batch, k]
                    elif n_models == 3:
                        # p1^4 / ((p1 + p2) * (p1^2 + p1*p3 + p2*p3))
                        # dominant_model_idx determines which model becomes p1 (dominant)
                        # Reorder indices so that dominant model is first
                        idx_order = [self.dominant_model_idx,
                                     (self.dominant_model_idx + 1) % 3,
                                     (self.dominant_model_idx + 2) % 3]
                        log_p1, log_p2, log_p3 = log_u_pf[idx_order[0]], log_u_pf[idx_order[1]], log_u_pf[idx_order[2]]

                        log_numerator = 4 * log_p1  # [n_batch, k]
                        # first term: p1 + p2
                        log_denom1 = torch.logsumexp(torch.stack([log_p1, log_p2], dim=0), dim=0)
                        # second term: p1^2 + p1*p3 + p2*p3
                        log_denom2 = torch.logsumexp(torch.stack([
                            2 * log_p1,           # p1^2
                            log_p1 + log_p3,      # p1*p3
                            log_p2 + log_p3       # p2*p3
                        ], dim=0), dim=0)
                        log_denominator = log_denom1 + log_denom2  # [n_batch, k]
                    
                    final = log_numerator - log_denominator  # [n_batch, k]
                    
                elif self.mode == "scalarization":
                    # Scalarization:
                    #   without_u=False: F_mix ∝ (Σ_i w_i · (pf_i · Z_i · u_i)^{1/β})^β
                    #   without_u=True:  F_mix ∝ Σ_i w_i · pf_i · Z_i
                    beta = 1.0 if self.without_u else self.beta
                    ws = [log_w_tensor[global_i, i] for i in range(n_models)]
                    log_Zs = [self.log_z_tensor[0, i] for i in range(n_models)]  # scalars
                    terms = []
                    for i, (pf, w) in enumerate(zip(fwds, ws)):
                        term = pf + log_Zs[i]
                        if not self.without_u:
                            # log u_i(s) = log F_i(s) - log Z_i, per active sample.
                            log_u_i = log_F_tensor[idx, i] - log_Zs[i]
                            term = term + log_u_i[:, None]
                        terms.append(w[:, None] + term / beta)
                    final = beta * torch.logsumexp(torch.stack(terms, dim=0), dim=0)  # [n, k]

                tmp.append(final)

            for i, logit in enumerate(tmp):
                guided_cat.logits[i] = logit

            # sample action from mixture distribution
            actions = guided_cat.sample()
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = guided_cat.log_prob(actions)

            # Per-model forward log probs for detailed-balance F update
            if self.use_db_f:
                per_model_log_probs = [fwd_cats[m].log_prob(actions) for m in range(n_models)]

            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                else:  # If not done, try to step the self.env
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]["is_valid"] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    data[i]["is_sink"].append(0)

                    # On-the-fly F update: log F(s') = log F(s) + log P_F(s'|s) + log(n_back)
                    if self.use_db_f:
                        log_n_back = math.log(n_back)
                        for m in range(n_models):
                            log_F_running[i, m] += per_model_log_probs[m][j].item() + log_n_back

                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can  construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]["is_valid"] = False
            if all(done):
                break

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1

        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]

        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                # TODO: instead of padding with Stop, we could have a virtual action whose
                # probability always evaluates to 1.
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data

def make_mixture_sampler_subtb(
    algo,
    ingredient_model_trainers,
    *,
    mode="scalarization",
    beta=32,
    dominant_model_idx=0,
    without_u=False,
    use_db_f=False,
):
    """Build a `MixtureGraphSampler` from an existing TB-style `algo`.

    Use this at eval time to swap `algo.graph_sampler`:

        algo.graph_sampler = make_mixture_sampler_subtb(algo, ingredient_trainers, mode=...)
    """
    return MixtureGraphSampler(
        algo.ctx,
        algo.env,
        algo.global_cfg.algo.max_len,
        algo.global_cfg.algo.max_nodes,
        algo.sample_temp,
        correct_idempotent=algo.cfg.do_correct_idempotent,
        pad_with_terminal_state=algo.cfg.do_parameterize_p_b,
        ingredient_model_trainers=ingredient_model_trainers,
        mode=mode,
        beta=beta,
        dominant_model_idx=dominant_model_idx,
        without_u=without_u,
        use_db_f=use_db_f,
    )
