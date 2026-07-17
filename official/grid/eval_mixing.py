import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from gfn.preprocessors import KHotPreprocessor

from src.utils import set_seed, save_plot_results, set_up_gflownet, get_exact_u_and_P_T
from src.advanced_hypergrid import AdvancedHyperGrid
from src.mixture_estimator import MixtureDiscretePolicyEstimator
from src.simplex import simplex_list


def evaluate(env, gflownets, preprocessor, n_reward_fns, n_simplex_for_val,
             mixing_type, loss, result_saved_path=None, beta=1.0,
             flow_estimation="logF"):

    env.set_train_status(False)

    no_simplex = mixing_type in ("harmonic_mean", "contrast",
                                  "harmonic_mean_without_u", "contrast_without_u")

    if no_simplex:
        n_models = len(gflownets)
        dummy_weights = [1.0 / n_models] * n_models

        with torch.no_grad():
            mixture_pf = MixtureDiscretePolicyEstimator(
                gfn_list=gflownets,
                n_actions=env.n_actions,
                preprocessor=preprocessor,
                loss_type=loss,
                is_backward=False,
                mixing_type=mixing_type,
                env=env,
                weight_list=dummy_weights,
                beta=beta,
                flow_estimation=flow_estimation,
            )

            true_dist = env.true_dist.cpu()
            _, learned_dist = get_exact_u_and_P_T(env, mixture_pf, cond=None)
            l1 = (learned_dist - true_dist).abs().sum().item()

        if result_saved_path:
            save_plot_results(
                env=env, saved_path=result_saved_path, pf_estimator=mixture_pf,
                best_l1=l1,
            )

        print(f"L1 Distance: {l1:.6f}")
        return {'mean_l1': l1, 'l1_errors': [l1]}

    # === Generate evenly spaced simplex points for evaluation ===
    simplex = simplex_list(n_reward_fns, n_simplex_for_val, env.device)

    print(f"Evaluating on {len(simplex)} simplex points...")

    # Pick 5 representative simplex points for visualization
    n_vis = min(5, len(simplex))
    vis_indices = np.linspace(0, len(simplex) - 1, n_vis, dtype=int).tolist()

    l1_errors = []
    vis_data = []  # (cond, true_dist, learned_dist, l1) for representative points

    with torch.no_grad():
        for i, cond in enumerate(tqdm(simplex)):
            mixture_pf = MixtureDiscretePolicyEstimator(
                gfn_list=gflownets,
                n_actions=env.n_actions,
                preprocessor=preprocessor,
                loss_type=loss,
                is_backward=False,
                mixing_type=mixing_type,
                env=env,
                weight_list=cond.squeeze(0).tolist(),
                beta=beta,
                flow_estimation=flow_estimation,
            )

            env.set_conditioning(cond)

            true_dist = env.true_dist.cpu()
            _, learned_dist = get_exact_u_and_P_T(env, mixture_pf, cond=None)

            l1 = (learned_dist - true_dist).abs().sum().item()
            l1_errors.append(l1)

            if i in vis_indices:
                vis_data.append((cond.squeeze(0).cpu(), true_dist, learned_dist, l1))

    mean_l1 = sum(l1_errors) / len(l1_errors)

    # Save visualization if requested
    if result_saved_path and vis_data:
        height = env.height
        n_vis = len(vis_data)
        fig, axes = plt.subplots(2, n_vis + 1, figsize=(4 * (n_vis + 1), 7))

        for col, (cond, td, ld, l1) in enumerate(vis_data):
            td_img = td.reshape(height, height).numpy()
            ld_img = ld.reshape(height, height).numpy()
            vmin = min(td_img.min(), ld_img.min())
            vmax = max(td_img.max(), ld_img.max())

            w_str = ",".join(f"{w:.2f}" for w in cond.tolist())

            im = axes[0, col].imshow(td_img, cmap="Blues", origin="upper",
                                     vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[0, col].set_title(f"True w=[{w_str}]", fontsize=9)
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])

            im = axes[1, col].imshow(ld_img, cmap="Blues", origin="upper",
                                     vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[1, col].set_title(f"Learned L1={l1:.4f}", fontsize=9)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])

        # Last column: L1 distribution + mean
        ax = axes[0, n_vis]
        ax.hist(l1_errors, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(mean_l1, color="red", linestyle="--", label=f"Mean={mean_l1:.4f}")
        ax.set_xlabel("L1")
        ax.set_title("L1 Distribution")
        ax.legend(fontsize=8)

        axes[1, n_vis].axis("off")
        axes[1, n_vis].text(0.5, 0.5, f"Mean L1: {mean_l1:.6f}\n"
                            f"Min L1: {min(l1_errors):.6f}\n"
                            f"Max L1: {max(l1_errors):.6f}\n"
                            f"N points: {len(l1_errors)}",
                            transform=axes[1, n_vis].transAxes, fontsize=12,
                            va="center", ha="center",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        plt.savefig(result_saved_path, dpi=150, bbox_inches="tight")
        plt.close()

    return {
        'mean_l1': mean_l1,
        'l1_errors': l1_errors,
    }


def main(args):
    set_seed(0)

    # === Initialize environment ===
    env = AdvancedHyperGrid(
        ndim=args.ndim,
        height=args.height,
        n_reward_fns=args.n_reward_fns,
        device=args.device,
        custom_dist=args.custom_dist,
        beta=args.beta,
    )

    # === Initialize preprocessor ===
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # === Load pre-trained GFlowNets ===
    gflownets = []
    ckpt_paths = args.model_saved_path_list.split(',')
    print(f"Loading {len(ckpt_paths)} pre-trained GFlowNets...")

    for ckpt_path in ckpt_paths:
        gfn = set_up_gflownet(
            env, args.hidden_dim, args.n_hidden, preprocessor,
            args.loss, args.subTB_weighting, args.subTB_lambda,
        ).to(args.device)

        gfn.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        gfn.eval()
        gflownets.append(gfn)

    # === Evaluate ===
    metrics = evaluate(
        env, gflownets, preprocessor,
        args.n_reward_fns, args.n_simplex_for_val,
        args.mixing_type, args.loss, args.result_saved_path, args.beta,
        args.flow_estimation,
    )

    print(f"\nEvaluation Results:")
    if len(metrics['l1_errors']) > 1:
        print(f"  Mean L1: {metrics['mean_l1']:.6f}")
        print(f"  Min L1:  {min(metrics['l1_errors']):.6f}")
        print(f"  Max L1:  {max(metrics['l1_errors']):.6f}")
    else:
        print(f"  L1: {metrics['mean_l1']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mixture of GFlowNets on Advanced HyperGrid")

    # Model checkpoints
    parser.add_argument("--model_saved_path_list", type=str, required=True,
                        help="Comma-separated list of paths to trained model checkpoints")

    # Environment configuration
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--custom_dist", type=str, default=None,
                        help="Reward function: mix2, mix3, etc.")
    parser.add_argument("--n_reward_fns", type=int, default=2,
                        help="Number of objectives (reward functions)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Reward sharpening exponent (must match training)")

    # Model architecture (must match training)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    # Mixing
    parser.add_argument("--mixing_type", type=str, required=True,
                        choices=["scalarization", "harmonic_mean", "contrast",
                                 "scalarization_without_u", "harmonic_mean_without_u", "contrast_without_u"])
    parser.add_argument("--flow_estimation", type=str, default="logF",
                        choices=["logF", "detailed_balance"],
                        help="How to estimate F(s): 'logF' reads from gfn.logF, "
                             "'detailed_balance' computes via DP (only matters for with-u variants)")

    # Output
    parser.add_argument("--result_saved_path", type=str, default=None,
                        help="Path to save evaluation plot")

    # Evaluation
    parser.add_argument("--n_simplex_for_val", type=int, default=128,
                        help="Number of evenly spaced simplex points for evaluation")

    args = parser.parse_args()
    main(args)
