import os
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from gfn.preprocessors import KHotPreprocessor
from src.utils import set_seed, set_up_gflownet, get_exact_u_and_P_T
from src.advanced_hypergrid import AdvancedHyperGrid
from src.classifier_utils import make_mlp, toin, compute_exact_logp


class JointYClassifierParam(nn.Module):
    """
    Classifier for 2-distribution composition with alpha parameterization.

    Outputs joint distribution Q(y1, y2 | s) for intermediate states
    and marginal distribution Q(y1 | x) for terminal states.
    """
    def __init__(self, height, ndim, num_hidden=256, num_layers=2):
        super().__init__()
        input_dim = ndim * height + 1  # one-hot coords + logit_alpha (masked for terminal)
        self.trunk = make_mlp([input_dim] + [num_hidden] * num_layers)
        self.non_term_head = nn.Linear(num_hidden + 1, 3)  # +1 for logit_alpha
        self.term_head = nn.Linear(num_hidden, 1)

    def get_outputs(self, x, logit_alpha, terminal):
        """
        Args:
            x: [batch_size, ndim * height] one-hot encoded coordinates
            logit_alpha: [batch_size] log(α/(1-α))
            terminal: [batch_size] 0.0 or 1.0
        """
        cond = logit_alpha * (1.0 - terminal)
        x = self.trunk(torch.cat((x, cond[:, None]), dim=1))

        non_term_outputs = self.non_term_head(torch.cat((x, cond[:, None]), dim=1))
        term_outputs = self.term_head(x)

        return non_term_outputs, term_outputs

    def forward(self, x, logit_alpha, terminal):
        """
        Returns log probabilities of joint distribution Q(y1, y2 | state).

        Shape: [batch_size, 2, 2] where [i, j] = log Q(y1=i+1, y2=j+1 | state)
        """
        non_term_outputs, term_outputs = self.get_outputs(x, logit_alpha, terminal)

        # Non-terminal: softmax over 4 joint outcomes
        non_term_tmp = torch.cat([non_term_outputs, torch.zeros_like(non_term_outputs[:, :1])], dim=1)
        non_term_log_probs = torch.log_softmax(non_term_tmp, dim=1)

        # Terminal: factorized p(y1) * p(y2|alpha)
        # p(y1=1) = sigmoid(-output), p(y1=2) = sigmoid(output)
        # p(y2=1|alpha) = sigmoid(-(output - logit_alpha)), p(y2=2|alpha) = sigmoid(output - logit_alpha)
        term_log_a = nn.functional.logsigmoid(-term_outputs)
        term_log_b = nn.functional.logsigmoid(term_outputs)
        term_log_c = nn.functional.logsigmoid(-(term_outputs - logit_alpha[:, None]))
        term_log_d = nn.functional.logsigmoid(term_outputs - logit_alpha[:, None])

        term_log_ab = torch.cat([term_log_a, term_log_b], dim=1)
        term_log_cd = torch.cat([term_log_c, term_log_d], dim=1)

        # p(y1, y2) = p(y1) * p(y2)
        term_log_probs = (term_log_ab[:, :, None] + term_log_cd[:, None, :]).view(-1, 4)

        log_probs = non_term_log_probs * (1.0 - terminal.view(-1, 1)) + term_log_probs * terminal.view(-1, 1)
        log_probs = log_probs.view(-1, 2, 2)

        return log_probs


INF = 1e9


@torch.no_grad()
def sample_from_model(gflownet, env, num_samples, device, return_trajectories=True):
    """Sample trajectories from a trained GFlowNet using the gfn library's sampling.

    Returns:
        terminal_states: [num_samples, ndim] tensor of terminal state coordinates
        all_states: [total_states, ndim] tensor of all visited states (non-terminal)
        traj_indices: [total_states] tensor mapping each state to its trajectory index
    """
    # Use the gflownet's built-in trajectory sampling
    trajectories = gflownet.sample_trajectories(
        env, n=num_samples, save_logprobs=False, save_estimator_outputs=False, epsilon=0.0
    )

    # trajectories.states has shape (max_length+1, n_trajectories, ndim)
    # trajectories.terminating_idx has shape (n_trajectories,) - step at which each traj terminated
    states_tensor = trajectories.states.tensor.cpu()  # Move to CPU for safer indexing
    terminating_idx = trajectories.terminating_idx.cpu()

    max_length_plus_1, n_traj, ndim = states_tensor.shape

    # Extract terminal states (the state just BEFORE the sink state)
    # terminating_idx points to the sink state [-1, -1], so actual terminal is at terminating_idx - 1
    terminal_states_list = []
    for i in range(n_traj):
        term_idx = terminating_idx[i].item() - 1  # -1 because terminating_idx points to sink
        terminal_states_list.append(states_tensor[term_idx, i])
    terminal_states = torch.stack(terminal_states_list).to(device)

    # Collect all non-terminal states along trajectories (excluding the sink state)
    all_states_list = []
    traj_indices_list = []

    for i in range(n_traj):
        traj_len = terminating_idx[i].item()  # This is the sink state index
        # Include all states from start to terminal (0 to traj_len-1, excluding sink)
        for t in range(traj_len):
            all_states_list.append(states_tensor[t, i])
            traj_indices_list.append(i)

    if len(all_states_list) > 0:
        all_states = torch.stack(all_states_list).to(device)
        traj_indices = torch.tensor(traj_indices_list, device=device)
    else:
        all_states = torch.zeros((0, ndim), device=device, dtype=torch.long)
        traj_indices = torch.zeros((0,), device=device, dtype=torch.long)

    return terminal_states, all_states, traj_indices


def get_guided_fwd_logits_fn(gflownet_1, gflownet_2, cls, height, ndim, device,
                              env_1, env_2, y1=1, y2=2, logit_alpha=0.0, just_mixture=False):
    """Create a guided forward logits function for classifier-guided sampling."""
    INF_VAL = 1e9

    def guided_fwd_logits_fn(z):
        z = z.to(device)
        enc = toin(z, height).to(device)

        with torch.no_grad():
            states_1 = env_1.States(z.clone().to(device))
            states_2 = env_2.States(z.clone().to(device))

            states_1.set_nonexit_action_masks(z.to(device) == height - 1, allow_exit=True)
            states_2.set_nonexit_action_masks(z.to(device) == height - 1, allow_exit=True)

            estimator_outputs_1 = gflownet_1.pf(states_1)
            estimator_outputs_2 = gflownet_2.pf(states_2)

            dist_1 = gflownet_1.pf.to_probability_distribution(states_1, estimator_outputs_1, epsilon=0.0)
            dist_2 = gflownet_2.pf.to_probability_distribution(states_2, estimator_outputs_2, epsilon=0.0)

            model_fwd_logprobs_1 = dist_1.logits
            model_fwd_logprobs_2 = dist_2.logits

        edge_mask = torch.cat([(z == height - 1).float(), torch.zeros((z.shape[0], 1), device=device)], dim=1)
        model_fwd_logprobs_1 = model_fwd_logprobs_1 - INF_VAL * edge_mask.to(device)
        model_fwd_logprobs_2 = model_fwd_logprobs_2 - INF_VAL * edge_mask.to(device)

        model_fwd_logprobs_1 = torch.log_softmax(model_fwd_logprobs_1, dim=1)
        model_fwd_logprobs_2 = torch.log_softmax(model_fwd_logprobs_2, dim=1)

        logit_alpha_tensor = torch.full((z.shape[0],), logit_alpha, device=device)

        cls_logprobs_cur = cls(enc.to(device), logit_alpha_tensor, torch.zeros(z.shape[0], device=device))

        logp_y1_eq_1_cur = torch.logsumexp(cls_logprobs_cur, dim=2)[:, 0]
        logp_y1_eq_2_cur = torch.logsumexp(cls_logprobs_cur, dim=2)[:, 1]

        mixture_logits = torch.logsumexp(
            torch.stack([model_fwd_logprobs_1 + logp_y1_eq_1_cur[:, None],
                         model_fwd_logprobs_2 + logp_y1_eq_2_cur[:, None]], dim=0),
            dim=0)

        if just_mixture:
            return mixture_logits

        z_next = z[:, None, :] + torch.eye(ndim, dtype=torch.long, device=device)[None, :, :]
        z_next_valid_mask = torch.all(z_next < height, dim=2)
        z_next = torch.minimum(z_next, torch.tensor(height - 1, device=z_next.device))
        z_next = z_next.view(-1, ndim)

        logit_alpha_tensor_next = torch.full((z_next.shape[0],), logit_alpha, device=device)

        cls_logprobs_next = cls(toin(z_next, height).to(device), logit_alpha_tensor_next,
                                torch.zeros(z_next.shape[0], device=device))
        cls_logprobs_next = cls_logprobs_next.view(z.shape[0], ndim, 2, 2)

        cls_logprobs_end = cls(enc.to(device), logit_alpha_tensor, torch.ones(z.shape[0], device=device))

        guidance_next = cls_logprobs_next[:, :, y1 - 1, y2 - 1] - cls_logprobs_cur[:, None, y1 - 1, y2 - 1]
        guidance_next[~z_next_valid_mask] = 0.0

        guidance_end = cls_logprobs_end[:, y1 - 1, y2 - 1] - cls_logprobs_cur[:, y1 - 1, y2 - 1]

        guidance = torch.cat([guidance_next, guidance_end[:, None]], dim=1)

        return mixture_logits + guidance

    return guided_fwd_logits_fn


def compute_ground_truth_distribution(logp_1, logp_2, alpha, mode='harmonic_mean'):
    """Compute ground truth composed distribution."""
    denom = torch.logsumexp(
        torch.stack([logp_1 + math.log(alpha), logp_2 + math.log(1 - alpha)], dim=0), dim=0)

    if mode == 'harmonic_mean':
        logp = logp_1 + logp_2 - denom
    elif mode == 'contrast_1':
        logp = logp_1 + logp_1 - denom
    elif mode == 'contrast_2':
        logp = logp_2 + logp_2 - denom
    elif mode == 'mixture':
        logp = torch.logsumexp(
            torch.stack([logp_1 + math.log(alpha), logp_2 + math.log(1 - alpha)], dim=0), dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.log_softmax(logp, dim=0)


def plot_training_status(gt_dists, model_dists, l1_errors, loss_history, height, step, save_path):
    """Plot training status: GT vs model for hm/contrast_1/contrast_2, plus loss curves.

    Args:
        gt_dists: dict of {operation: numpy_array} ground truth distributions
        model_dists: dict of {operation: numpy_array} model distributions (can be None)
        l1_errors: dict of {operation: float} L1 errors (can be None)
        loss_history: dict with 'total', 'term', 'non_term' lists
        height: grid height
        step: current training step
        save_path: path to save figure
    """
    op_names = {
        'harmonic_mean': 'Harmonic Mean',
        'contrast_1': 'Contrast 1',
        'contrast_2': 'Contrast 2',
    }
    operations = ['harmonic_mean', 'contrast_1', 'contrast_2']

    # Layout: 2 rows x 3 cols for operations + 1 loss plot
    # Row 0: GT distributions | Row 1: Model distributions
    # Col 3 (spanning rows): loss curves
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])

    for col, op in enumerate(operations):
        gt_img = gt_dists[op].reshape(height, height)

        # Determine color scale from GT
        vmin, vmax = gt_img.min(), gt_img.max()
        if model_dists and op in model_dists and model_dists[op] is not None:
            model_img = model_dists[op].reshape(height, height)
            vmin = min(vmin, model_img.min())
            vmax = max(vmax, model_img.max())

        # Top row: GT
        ax_gt = fig.add_subplot(gs[0, col])
        im = ax_gt.imshow(gt_img, origin='upper', cmap='Blues', vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
        ax_gt.set_title(f'GT: {op_names[op]}', fontsize=10)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        plt.colorbar(im, ax=ax_gt, fraction=0.046, pad=0.04)

        # Bottom row: Model
        ax_model = fig.add_subplot(gs[1, col])
        if model_dists and op in model_dists and model_dists[op] is not None:
            model_img = model_dists[op].reshape(height, height)
            im = ax_model.imshow(model_img, origin='upper', cmap='Blues', vmin=vmin, vmax=vmax,
                                 interpolation='nearest', aspect='equal')
            title = f'Model (Step {step})'
            if l1_errors and op in l1_errors:
                title += f'\nL1: {l1_errors[op]:.4f}'
            ax_model.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax_model, fraction=0.046, pad=0.04)
        else:
            ax_model.text(0.5, 0.5, 'Not computed', ha='center', va='center', transform=ax_model.transAxes)
            ax_model.set_title(f'Model (Step {step})', fontsize=10)
        ax_model.set_xticks([])
        ax_model.set_yticks([])

    # Loss curves (right column, spanning both rows)
    ax_loss = fig.add_subplot(gs[:, 3])
    steps = list(range(len(loss_history['total'])))
    ax_loss.plot(steps, loss_history['total'], label='Total Loss', linewidth=2)
    ax_loss.plot(steps, loss_history['term'], label='Terminal Loss', alpha=0.7)
    ax_loss.plot(steps, loss_history['non_term'], label='Non-terminal Loss', alpha=0.7)
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train(args):
    set_seed(args.seed)
    device = args.device

    # Load pretrained GFlowNets
    print(f"Loading model 1 from {args.model_path_1}")
    print(f"Loading model 2 from {args.model_path_2}")

    # Initialize environments for both models
    env_1 = AdvancedHyperGrid(
        ndim=args.ndim, height=args.height, device=device,
        custom_dist=args.custom_dist_1,
    )
    env_2 = AdvancedHyperGrid(
        ndim=args.ndim, height=args.height, device=device,
        custom_dist=args.custom_dist_2,
    )

    preprocessor = KHotPreprocessor(height=args.height, ndim=args.ndim)

    # Set up GFlowNets
    gflownet_1 = set_up_gflownet(
        env_1, args.hidden_dim, args.n_hidden, preprocessor,
        args.loss, args.subTB_weighting, args.subTB_lambda,
    ).to(device)
    gflownet_2 = set_up_gflownet(
        env_2, args.hidden_dim, args.n_hidden, preprocessor,
        args.loss, args.subTB_weighting, args.subTB_lambda,
    ).to(device)

    gflownet_1.load_state_dict(torch.load(args.model_path_1, map_location='cpu', weights_only=True))
    gflownet_2.load_state_dict(torch.load(args.model_path_2, map_location='cpu', weights_only=True))
    gflownet_1.eval()
    gflownet_2.eval()

    # Initialize classifier
    cls = JointYClassifierParam(
        height=args.height, ndim=args.ndim,
        num_hidden=args.cls_hidden_dim, num_layers=args.cls_n_layers,
    ).to(device)

    target_cls = copy.deepcopy(cls)
    for p in target_cls.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(cls.parameters(), lr=args.lr)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    save_prefix = f"{args.custom_dist_1}_{args.custom_dist_2}_seed{args.seed}"

    # Compute ground truth distributions for all 3 operations at alpha=0.5
    print("Computing ground truth distributions for visualization...")
    _, logp_1 = get_exact_u_and_P_T(env_1, gflownet_1.pf)
    _, logp_2 = get_exact_u_and_P_T(env_2, gflownet_2.pf)
    logp_1 = torch.log(logp_1 + 1e-40).to(device)
    logp_2 = torch.log(logp_2 + 1e-40).to(device)

    gt_dists = {}
    for op in ['harmonic_mean', 'contrast_1', 'contrast_2']:
        gt_logp = compute_ground_truth_distribution(logp_1, logp_2, alpha=0.5, mode=op)
        gt_dists[op] = torch.exp(gt_logp).detach().cpu().numpy()

    # y1, y2 settings for each operation
    op_y_settings = {
        'harmonic_mean': (1, 2),
        'contrast_1': (1, 1),
        'contrast_2': (2, 2),
    }

    # Loss history for plotting
    loss_history = {'total': [], 'term': [], 'non_term': []}

    # Training loop
    for step in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        # Sample trajectories from both models
        x_1, all_states_1, traj_idx_1 = sample_from_model(gflownet_1, env_1, args.batch_size, device)
        x_2, all_states_2, traj_idx_2 = sample_from_model(gflownet_2, env_2, args.batch_size, device)

        # Sample alpha
        u = torch.rand(2 * args.batch_size, device=device)
        logit_alpha = args.logit_alpha_min + (args.logit_alpha_max - args.logit_alpha_min) * u

        # Terminal loss
        x_term = torch.cat([x_1, x_2], dim=0)
        ce_target_term = torch.cat([
            torch.zeros(x_1.shape[0], device=device),
            torch.ones(x_2.shape[0], device=device)
        ], dim=0)

        enc_term = toin(x_term, args.height).to(device)
        logprobs_term = cls(enc_term, logit_alpha, torch.ones(enc_term.shape[0], device=device))

        log_p_y_eq_1 = torch.logsumexp(logprobs_term, dim=2)[:, 0]
        log_p_y_eq_2 = torch.logsumexp(logprobs_term, dim=2)[:, 1]

        loss_term = -torch.mean(ce_target_term * log_p_y_eq_2 + (1.0 - ce_target_term) * log_p_y_eq_1)

        # Non-terminal loss
        s_non_term = torch.cat([all_states_1, all_states_2], dim=0)
        enc_non_term = toin(s_non_term, args.height).to(device)

        # Build trajectory indices for logit_alpha
        traj_lens = torch.tensor(
            [len(all_states_1[traj_idx_1 == i]) for i in range(args.batch_size)] +
            [len(all_states_2[traj_idx_2 == i]) for i in range(args.batch_size)],
            device=device
        )
        traj_ind = torch.arange(0, traj_lens.shape[0], device=device).repeat_interleave(traj_lens)

        # Get target p(y2|x) from EMA network
        with torch.no_grad():
            _, term_outputs_ema = target_cls.get_outputs(
                enc_term, logit_alpha,
                torch.ones(enc_term.shape[0], device=device)
            )
            p_x_y2_eq_1 = torch.sigmoid(-(term_outputs_ema - logit_alpha[:, None])).squeeze()
            p_x_y2_eq_2 = torch.sigmoid(term_outputs_ema - logit_alpha[:, None]).squeeze()

        logprobs_non_term = cls(enc_non_term, logit_alpha[traj_ind], torch.zeros(enc_non_term.shape[0], device=device))

        w_s_y2_eq_1 = p_x_y2_eq_1[traj_ind]
        w_s_y2_eq_2 = p_x_y2_eq_2[traj_ind]

        # Build weight matrix
        w_mat = torch.zeros((s_non_term.shape[0], 2, 2), device=device)
        n_states_1 = all_states_1.shape[0]
        # y1=1 for model 1
        w_mat[:n_states_1, 0, 0] = 1.0
        w_mat[:n_states_1, 0, 1] = 1.0
        # y1=2 for model 2
        w_mat[n_states_1:, 1, 0] = 1.0
        w_mat[n_states_1:, 1, 1] = 1.0

        w_mat[:, :, 0] *= w_s_y2_eq_1[:, None]
        w_mat[:, :, 1] *= w_s_y2_eq_2[:, None]

        loss_non_term = -torch.sum(w_mat * logprobs_non_term) / (2 * args.batch_size)

        # Gamma warmup
        gamma = min(1.0, step / args.loss_non_term_weight_steps) if args.loss_non_term_weight_steps > 0 else 1.0

        loss = loss_term + loss_non_term * gamma

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network with EMA
        for a, b in zip(cls.parameters(), target_cls.parameters()):
            b.data.mul_(args.target_network_ema).add_(a.data * (1 - args.target_network_ema))

        # Track loss history
        loss_history['total'].append(loss.item())
        loss_history['term'].append(loss_term.item())
        loss_history['non_term'].append(loss_non_term.item())

        # Save checkpoint and visualization
        if (step + 1) % args.save_every == 0 or step == args.n_iterations - 1:
            cls.eval()
            try:
                logit_alpha_vis = 0.0  # alpha = 0.5
                model_dists = {}
                l1_errors = {}

                for op, (y1, y2) in op_y_settings.items():
                    guided_fwd_fn = get_guided_fwd_logits_fn(
                        gflownet_1, gflownet_2, cls, args.height, args.ndim, device,
                        env_1, env_2, y1=y1, y2=y2, logit_alpha=logit_alpha_vis,
                        just_mixture=False,
                    )
                    model_logp = compute_exact_logp(guided_fwd_fn, args.height, args.ndim, device)
                    model_dist = torch.exp(model_logp)
                    model_dist = (model_dist / model_dist.sum()).detach().cpu().numpy()
                    model_dists[op] = model_dist
                    l1_errors[op] = np.abs(model_dist - gt_dists[op]).sum()

                vis_path = os.path.join(args.save_dir, f'{save_prefix}.png')
                plot_training_status(gt_dists, model_dists, l1_errors, loss_history, args.height, step + 1, vis_path)
                hm_l1 = l1_errors['harmonic_mean']
                print(f"  Vis saved: {vis_path} | hm L1: {hm_l1:.4f}, c1 L1: {l1_errors['contrast_1']:.4f}, c2 L1: {l1_errors['contrast_2']:.4f}")

            except Exception as e:
                print(f"  Visualization/evaluation failed: {e}")
                hm_l1 = None
            cls.train()

            # Save model at last step
            if step == args.n_iterations - 1:
                ckpt_path = os.path.join(args.save_dir, f'{save_prefix}_{step + 1}steps.pt')
                torch.save({
                    'cls': cls.state_dict(),
                    'target_cls': target_cls.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'args': vars(args),
                }, ckpt_path)
                print(f"  Model saved: {ckpt_path}")

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "term": f"{loss_term.item():.4f}",
            "non_term": f"{loss_non_term.item():.4f}",
            "gamma": f"{gamma:.3f}",
        })


def main():
    parser = argparse.ArgumentParser(description="Train classifier for GFlowNet composition on HyperGrid")

    # Model paths
    parser.add_argument("--model_path_1", type=str, required=True, help="Path to first pretrained GFlowNet")
    parser.add_argument("--model_path_2", type=str, required=True, help="Path to second pretrained GFlowNet")

    # Environment
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--custom_dist_1", type=str, default=None, help="Custom distribution for model 1")
    parser.add_argument("--custom_dist_2", type=str, default=None, help="Custom distribution for model 2")

    # GFlowNet model (for loading pretrained models)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    # Classifier
    parser.add_argument("--cls_hidden_dim", type=int, default=64, help="Hidden dim for classifier")
    parser.add_argument("--cls_n_layers", type=int, default=2, help="Number of hidden layers for classifier")

    # Training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_iterations", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--target_network_ema", type=float, default=0.995)
    parser.add_argument("--loss_non_term_weight_steps", type=int, default=3000, help="Steps for gamma warmup")
    parser.add_argument("--logit_alpha_min", type=float, default=-3.5)
    parser.add_argument("--logit_alpha_max", type=float, default=3.5)

    # Saving
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoint and progress plot")
    parser.add_argument("--save_every", type=int, default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
