import math
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from gfn.preprocessors import KHotPreprocessor
from src.utils import set_seed, set_up_gflownet, get_exact_u_and_P_T
from src.advanced_hypergrid import AdvancedHyperGrid
from src.classifier_utils import compute_exact_logp, toin
from train_classifier_2way import JointYClassifierParam


def get_guided_fwd_logits_fn(gflownet_1, gflownet_2, cls, height, ndim, device,
                              preprocessor, env_1, env_2, y1=1, y2=2, logit_alpha=0.0, just_mixture=False):
    """Create a guided forward logits function for classifier-guided sampling."""

    INF = 1e9

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
        model_fwd_logprobs_1 = model_fwd_logprobs_1 - INF * edge_mask.to(device)
        model_fwd_logprobs_2 = model_fwd_logprobs_2 - INF * edge_mask.to(device)

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
        logp = denom
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.log_softmax(logp, dim=0)


def evaluate(args):
    set_seed(args.seed)
    device = args.device

    # Load environments
    env_1 = AdvancedHyperGrid(ndim=args.ndim, height=args.height, device=device, custom_dist=args.custom_dist_1)
    env_2 = AdvancedHyperGrid(ndim=args.ndim, height=args.height, device=device, custom_dist=args.custom_dist_2)

    preprocessor = KHotPreprocessor(height=args.height, ndim=args.ndim)

    # Load GFlowNets
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

    # Load classifier
    cls = JointYClassifierParam(
        height=args.height, ndim=args.ndim,
        num_hidden=args.cls_hidden_dim, num_layers=args.cls_n_layers,
    ).to(device)

    checkpoint = torch.load(args.cls_path, weights_only=False)
    cls.load_state_dict(checkpoint['cls'])
    cls.eval()

    # Compute base distributions
    print("Computing base distributions...")
    _, logp_1 = get_exact_u_and_P_T(env_1, gflownet_1.pf)
    _, logp_2 = get_exact_u_and_P_T(env_2, gflownet_2.pf)
    logp_1 = torch.log(logp_1 + 1e-40).to(device)
    logp_2 = torch.log(logp_2 + 1e-40).to(device)

    # Evaluate HM, CT1, CT2
    alpha = 0.5
    logit_alpha = 0.0
    op_settings = [
        ('Harmonic Mean', 'harmonic_mean', 1, 2),
        ('Contrast 1', 'contrast_1', 1, 1),
        ('Contrast 2', 'contrast_2', 2, 2),
    ]

    results = {}
    gt_dists = {}
    model_dists = {}

    for op_name, mode, y1, y2 in op_settings:
        print(f"  Evaluating {op_name}...")
        gt_logp = compute_ground_truth_distribution(logp_1, logp_2, alpha, mode)
        gt_dist = torch.exp(gt_logp).detach().cpu().numpy()
        gt_dists[mode] = gt_dist

        guided_fwd_logits_fn = get_guided_fwd_logits_fn(
            gflownet_1, gflownet_2, cls, args.height, args.ndim, device,
            preprocessor, env_1, env_2, y1=y1, y2=y2, logit_alpha=logit_alpha,
        )

        model_logp = compute_exact_logp(guided_fwd_logits_fn, args.height, args.ndim, device)
        model_dist = torch.exp(model_logp)
        model_dist = (model_dist / model_dist.sum()).detach().cpu().numpy()
        model_dists[mode] = model_dist

        l1 = np.abs(model_dist - gt_dist).sum()
        results[mode] = l1
        print(f"    L1: {l1:.6f}")

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for mode, l1 in results.items():
        print(f"  {mode}: L1={l1:.6f}")

    # Plot: 2 rows x 3 cols, GT on top, Model on bottom, shared colorscale per column
    if args.result_saved_path:
        height = args.height
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        for col, (op_name, mode, _, _) in enumerate(op_settings):
            gt_img = gt_dists[mode].reshape(height, height)
            model_img = model_dists[mode].reshape(height, height)
            vmin = min(gt_img.min(), model_img.min())
            vmax = max(gt_img.max(), model_img.max())

            im = axes[0, col].imshow(gt_img, origin='upper', cmap='Blues',
                                     vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
            axes[0, col].set_title(f'GT: {op_name}', fontsize=11)
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

            im = axes[1, col].imshow(model_img, origin='upper', cmap='Blues',
                                     vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
            axes[1, col].set_title(f'Model: L1={results[mode]:.4f}', fontsize=11)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

        plt.suptitle(f'{args.custom_dist_1} + {args.custom_dist_2}', fontsize=13)
        plt.tight_layout()
        plt.savefig(args.result_saved_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {args.result_saved_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier-guided GFlowNet composition")

    # Model paths
    parser.add_argument("--model_path_1", type=str, required=True)
    parser.add_argument("--model_path_2", type=str, required=True)
    parser.add_argument("--cls_path", type=str, required=True, help="Path to trained classifier")

    # Environment
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--custom_dist_1", type=str, required=True)
    parser.add_argument("--custom_dist_2", type=str, required=True)

    # GFlowNet model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    # Classifier
    parser.add_argument("--cls_hidden_dim", type=int, default=64)
    parser.add_argument("--cls_n_layers", type=int, default=2)

    # Evaluation
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result_saved_path", type=str, default=None,
                        help="Path to save evaluation plot (single png)")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
