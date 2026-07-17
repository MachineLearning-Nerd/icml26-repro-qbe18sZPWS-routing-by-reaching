import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from gfn.gym import HyperGrid
from gfn.estimators import Estimator, ScalarEstimator
from gfn.utils.modules import MLP
from gfn.gflownet import SubTBGFlowNet
from gfn.estimators import (
    ConditionalDiscretePolicyEstimator,
    DiscretePolicyEstimator,
    ConditionalScalarEstimator,
)
from src.hypernetwork import HyperNetwork, HNDiscretePolicyEstimator, HNScalarEstimator


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_exact_u_and_P_T(env: HyperGrid, pf_estimator: Estimator, cond: torch.Tensor = None) -> torch.Tensor:
    r"""Exact terminating-state distribution P_T for 2D HyperGrid
    Evaluates the exact terminating state distribution P_T for HyperGrid.

    For each state s', the terminating state probability is computed as:

    .. math::
        P_T(s') = u(s') P_F(s_f | s')

    where u(s') satisfies the recursion:

    .. math::
        u(s') = \sum_{s \in \text{Par}(s')} u(s) P_F(s' | s)

    with the base case u(s_0) = 1.

    Args:
        env: The HyperGrid environment
        pf: The GFlowNet's pf estimator

    Returns:
        The exact terminating state distribution as a tensor
    """

    if env.ndim != 2:
        raise ValueError("plotting is only supported for 2D environments")

    grid = env.all_states  # grid.tensor: (N, 2) integer coords

    # Not allowed to take any action beyond the environment height, but
    # allow early termination.
    grid.set_nonexit_action_masks(
        grid.tensor == env.height - 1,
        allow_exit=True,
    )

    with torch.no_grad():
        # forward policy over all states (GPU if available)
        grid.tensor = grid.tensor.to(env.device)

        if cond is not None:
            estimator_outputs = pf_estimator(grid, cond.expand(grid.tensor.shape[0],-1))
        else:
            estimator_outputs = pf_estimator(grid)
        dist = pf_estimator.to_probability_distribution(
            states=grid, 
            module_output=estimator_outputs,
            epsilon=0.0,
        )
        probs = dist.probs.detach().cpu()     # (N, A) normalized action probs
        grid.tensor = grid.tensor.cpu()
        
    coords = grid.tensor.long()               # (N, 2) on CPU
    N = coords.size(0)

    # Build coord -> flat index map for O(1) lookups
    Hx = int(coords[:, 0].max().item()) + 1
    Hy = int(coords[:, 1].max().item()) + 1
    coord_map = -torch.ones(Hx, Hy, dtype=torch.long)
    coord_map[coords[:, 0], coords[:, 1]] = torch.arange(N, dtype=torch.long)

    # u(s0)=1, others 0
    u = torch.zeros(N)
    s0_flat = coord_map[0, 0].item()
    u[s0_flat] = 1.0

    # Topological order over states (assumes env.all_indices() yields lexicographic)
    indices = env.all_indices()  # list of tuples like (i, j)
    for index in indices:
        if index == (0, 0):
            continue

        # parents: decrement one coordinate; last column = which action (0 or 1)
        parents = [
            ( (index[0]-1, index[1]), 0 ) if index[0] > 0 else None,
            ( (index[0],   index[1]-1), 1 ) if index[1] > 0 else None,
        ]
        parents = [p for p in parents if p is not None]
        if not parents:
            continue

        # Vectorized parent lookups
        p_coords = torch.tensor([p[0] for p in parents], dtype=torch.long)
        a_idx    = torch.tensor([p[1] for p in parents], dtype=torch.long)
        p_flat   = coord_map[p_coords[:, 0], p_coords[:, 1]]      # (P,)
        u_vals   = u[p_flat]                                      # (P,)
        p_probs  = probs[p_flat, a_idx]                           # (P,)

        cur_flat = coord_map[index[0], index[1]].item()
        u[cur_flat] = (u_vals * p_probs).sum()

    # Termination probability is action "-1" (last action) at each state
    P_T = (u * probs[:, -1]).detach().cpu() # If you don't want to force to stop at edge
    return u, P_T


def compute_log_F_by_detailed_balance(env, gfn):
    """Compute log F(s) for all states via detailed balance DP.

    Uses detailed balance:
        log F(s') = log F(s) + log P_F(s' | s) - log P_B(s | s')
    starting from log F(s_0).

    For SubTB models, log F(s_0) comes from `gfn.logF(s_0)`. Final log F(s) is
    averaged (logsumexp - log(2)) over all parent estimates when multiple parents.

    Returns:
        log_F tensor of shape [N] on CPU.
    """
    import math

    device = env.device
    grid = env.all_states
    grid.tensor = grid.tensor.to(device)
    grid.set_nonexit_action_masks(grid.tensor == env.height - 1, allow_exit=True)

    with torch.no_grad():
        pf_out = gfn.pf(grid)
        pf_dist = gfn.pf.to_probability_distribution(states=grid, module_output=pf_out, epsilon=0.0)
        fwd_log_probs = pf_dist.probs.clamp(min=1e-30).log().cpu()

        pb_out = gfn.pb(grid)
        pb_dist = gfn.pb.to_probability_distribution(states=grid, module_output=pb_out, epsilon=0.0)
        bwd_log_probs = pb_dist.probs.clamp(min=1e-30).log().cpu()

    grid.tensor = grid.tensor.cpu()
    coords = grid.tensor.long()
    N = coords.size(0)
    H = env.height

    coord_map = -torch.ones(H, H, dtype=torch.long)
    coord_map[coords[:, 0], coords[:, 1]] = torch.arange(N, dtype=torch.long)

    log_F = torch.full((N,), float('-inf'))
    s0_flat = coord_map[0, 0].item()
    s0_state = env.states_from_tensor(env.s0)
    log_F[s0_flat] = gfn.logF(s0_state).item()

    for index in env.all_indices():
        if index == (0, 0):
            continue
        i, j = index
        estimates = []

        if i > 0:
            p_flat = coord_map[i - 1, j].item()
            est = log_F[p_flat] + fwd_log_probs[p_flat, 0] - bwd_log_probs[coord_map[i, j].item(), 0]
            estimates.append(est)
        if j > 0:
            p_flat = coord_map[i, j - 1].item()
            est = log_F[p_flat] + fwd_log_probs[p_flat, 1] - bwd_log_probs[coord_map[i, j].item(), 1]
            estimates.append(est)

        cur_flat = coord_map[i, j].item()
        if len(estimates) == 1:
            log_F[cur_flat] = estimates[0]
        else:
            log_F[cur_flat] = torch.logsumexp(torch.stack(estimates), dim=0) - math.log(2)

    return log_F


def save_plot_results(env, saved_path, pf_estimator=None, l1_distances=None,
                      validation_interval=None, cond=None,
                      best_l1=None, learned_Z=None, true_Z=None):
    """Save visualization of true vs learned distribution.

    Args:
        env: The environment
        saved_path: Path to save the plot
        pf_estimator: Forward policy estimator
        l1_distances: List of L1 distances over training (optional)
        validation_interval: Steps between validations (for x-axis)
        cond: Conditioning tensor (optional)
        best_l1: Best L1 distance so far
        learned_Z: Learned partition function of best model
        true_Z: True partition function
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # === Get normalized distributions ===
    true_dist = env.true_dist.reshape(env.height, env.height).cpu().numpy()
    _, learned_dist = get_exact_u_and_P_T(env, pf_estimator, cond)
    learned_dist = learned_dist.reshape(env.height, env.height).numpy()

    # Shared colorbar range
    vmin = min(true_dist.min(), learned_dist.min())
    vmax = max(true_dist.max(), learned_dist.max())

    # === True distribution ===
    im0 = axes[0, 0].imshow(
        true_dist, cmap="Blues", interpolation="nearest",
        origin="upper", vmin=vmin, vmax=vmax, aspect="equal",
    )
    axes[0, 0].set_title("True Distribution")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im0, ax=axes[0, 0])

    # === Learned distribution ===
    im1 = axes[0, 1].imshow(
        learned_dist, cmap="Blues", interpolation="nearest",
        origin="upper", vmin=vmin, vmax=vmax, aspect="equal",
    )
    axes[0, 1].set_title("Learned Distribution")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 1])

    # === L1 progress ===
    if l1_distances is not None and len(l1_distances) > 0:
        steps = [(i + 1) * validation_interval for i in range(len(l1_distances))]
        axes[1, 0].plot(steps, l1_distances, linewidth=2)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("L1 Error")
        axes[1, 0].set_title("L1 Error Evolution")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    # === Metrics text ===
    axes[1, 1].axis("off")
    metrics_text = ""
    if best_l1 is not None:
        metrics_text += f"Best L1: {best_l1:.6f}\n"
    if true_Z is not None:
        metrics_text += f"Z_true: {true_Z:.4f}\n"
    if learned_Z is not None:
        metrics_text += f"Best Z_learned: {learned_Z:.4f}\n"
    if metrics_text:
        axes[1, 1].text(
            0.5, 0.5, metrics_text,
            transform=axes[1, 1].transAxes,
            fontsize=14, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(saved_path, dpi=150, bbox_inches="tight")
    plt.close()


def set_up_logF_estimator(hidden_dim, n_hidden, preprocessor, pf_module=None, trunk=None):
    """Returns a LogStateFlowEstimator."""
    module = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=1,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
        trunk=None,
    )
    return ScalarEstimator(module=module, preprocessor=preprocessor)


def set_up_gflownet(env, hidden_dim, n_hidden, preprocessor, loss,
                    subTB_weighting=None, subTB_lambda=None,
                    logF_n_hidden=None):
    # Build the pf, pb estimator.
    pf_module = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
    )
    pb_module = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
    )

    pf_estimator = DiscretePolicyEstimator(
        pf_module,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        pb_module,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True
    )

    logF_estimator = set_up_logF_estimator(
        hidden_dim=hidden_dim,
        n_hidden=logF_n_hidden if logF_n_hidden is not None else n_hidden,
        preprocessor=preprocessor,
    )
    return SubTBGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
        logF=logF_estimator,
        weighting=subTB_weighting,
        lamda=subTB_lambda,
    )


def build_conditional_pf_pb(env, preprocessor, n_reward_fns,
                            hidden_dim, n_hidden, cond_hidden_dim=16):
    # State encoder
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,    
    )
    module_PB = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
    )

    # Conditioning encoder (smaller than state encoder)
    module_cond = MLP(
        input_dim=n_reward_fns,
        output_dim=cond_hidden_dim,
        hidden_dim=cond_hidden_dim,
        n_hidden_layers=n_hidden,
    )

    # Modules post-concatenation
    concat_dim = hidden_dim + cond_hidden_dim
    module_final_PF = MLP(
        input_dim=concat_dim,
        output_dim=env.n_actions,
        hidden_dim=concat_dim,
        n_hidden_layers=n_hidden,
    )
    module_final_PB = MLP(
        input_dim=concat_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=concat_dim,
        n_hidden_layers=n_hidden,
    )

    pf_estimator = ConditionalDiscretePolicyEstimator(
        module_PF,
        module_cond,
        module_final_PF,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=False,
    )
    pb_estimator = ConditionalDiscretePolicyEstimator(
        module_PB,
        module_cond,
        module_final_PB,
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
    )

    return pf_estimator, pb_estimator


def build_conditional_logF_scalar_estimator(preprocessor, n_reward_fns, hidden_dim,
                                             n_hidden=2, cond_hidden_dim=16):
    # State encoder
    module_state_logF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
    )
    # Conditioning encoder (smaller than state encoder)
    module_conditioning_logF = MLP(
        input_dim=n_reward_fns,
        output_dim=cond_hidden_dim,
        hidden_dim=cond_hidden_dim,
        n_hidden_layers=n_hidden,
    )
    # Final module after concatenation
    concat_dim = hidden_dim + cond_hidden_dim
    module_final_logF = MLP(
        input_dim=concat_dim,
        output_dim=1,
        hidden_dim=concat_dim,
        n_hidden_layers=n_hidden,
    )

    logF_estimator = ConditionalScalarEstimator(
        module_state_logF,
        module_conditioning_logF,
        module_final_logF,
        preprocessor=preprocessor,
    )

    return logF_estimator


def set_up_conditional_gflownet(env, hidden_dim, n_hidden, preprocessor, loss,
                                n_reward_fns, cond_hidden_dim=16,
                                subTB_weighting=None, subTB_lambda=None):
    pf_estimator, pb_estimator = build_conditional_pf_pb(
        env=env,
        preprocessor=preprocessor,
        n_reward_fns=n_reward_fns,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        cond_hidden_dim=cond_hidden_dim,
    )
    logF_estimator = build_conditional_logF_scalar_estimator(
        preprocessor=preprocessor,
        n_reward_fns=n_reward_fns,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        cond_hidden_dim=cond_hidden_dim,
    )

    gflownet = SubTBGFlowNet(
        logF=logF_estimator,
        pf=pf_estimator,
        pb=pb_estimator,
        weighting=subTB_weighting,
        lamda=subTB_lambda,
    )

    return gflownet


def set_up_hn_gflownet(env, preprocessor, n_reward_fns, ray_hidden_dim, hidden_dim, n_hidden,
                        subTB_weighting=None, subTB_lambda=None,
                        predictor_layers=None, logit_clipping=0.0):
    """Set up HyperNetwork-based GFlowNet (HN-GFN).

    Creates HN-GFN following the original paper's architecture:
    - Fixed encoder: Shared MLP that encodes states (trained via backprop)
    - HyperNetwork: Generates multi-layer predictor head weights from preference vectors
    - LeakyReLU activations between generated layers
    - Optional logit clipping for numerical stability

    Args:
        env: The HyperGrid environment
        preprocessor: Preprocessor for state encoding
        n_reward_fns: Number of reward functions (objectives)
        ray_hidden_dim: Hidden dimension for HyperNetwork's ray MLP
        hidden_dim: Hidden dimension for fixed encoder
        n_hidden: Number of hidden layers in fixed encoder
        subTB_weighting: Weighting scheme for SubTB loss
        subTB_lambda: Lambda parameter for SubTB loss
        predictor_layers: List of hidden dimensions for generated predictor head.
                         e.g., [64, 64] means: encoder_output_dim -> 64 -> 64 -> n_actions
                         Default: [hidden_dim] (one hidden layer, matching original HN-GFN)
        logit_clipping: If > 0, apply tanh-based logit clipping for numerical stability

    Returns:
        SubTBGFlowNet with HyperNetwork-based estimators
    """
    state_dim = preprocessor.output_dim

    # Default predictor layers: one hidden layer with hidden_dim units (matching original HN-GFN)
    if predictor_layers is None:
        predictor_layers = [hidden_dim]

    # === Fixed Encoders (trained via backprop, shared across all preferences) ===
    # Forward policy encoder
    encoder_pf = MLP(
        input_dim=state_dim,
        output_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,    )

    # Backward policy encoder
    encoder_pb = MLP(
        input_dim=state_dim,
        output_dim=hidden_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
    )

    # === HyperNetworks (generate multi-layer predictor head weights) ===
    # Forward policy hypernetwork
    hn_pf = HyperNetwork(
        n_objectives=n_reward_fns,
        ray_hidden_dim=ray_hidden_dim,
        encoder_output_dim=hidden_dim,
        n_actions=env.n_actions,
        predictor_layers=predictor_layers,
        logit_clipping=logit_clipping,
    )

    # Backward policy hypernetwork
    hn_pb = HyperNetwork(
        n_objectives=n_reward_fns,
        ray_hidden_dim=ray_hidden_dim,
        encoder_output_dim=hidden_dim,
        n_actions=env.n_actions - 1,
        predictor_layers=predictor_layers,
        logit_clipping=logit_clipping,
    )

    # === Create HyperNetwork-based estimators ===
    pf_estimator = HNDiscretePolicyEstimator(
        encoder=encoder_pf,
        hypernetwork=hn_pf,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
        is_backward=False,
    )

    pb_estimator = HNDiscretePolicyEstimator(
        encoder=encoder_pb,
        hypernetwork=hn_pb,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
    )

    # logF: concat-based approach (matching original HN-GFN)
    logF_estimator = build_conditional_logF_scalar_estimator(
        preprocessor=preprocessor,
        n_reward_fns=n_reward_fns,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
    )

    # Create SubTBGFlowNet
    gflownet = SubTBGFlowNet(
        logF=logF_estimator,
        pf=pf_estimator,
        pb=pb_estimator,
        weighting=subTB_weighting,
        lamda=subTB_lambda,
    )

    return gflownet
