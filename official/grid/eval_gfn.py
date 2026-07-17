import argparse

import torch

from gfn.preprocessors import KHotPreprocessor

from src.utils import set_seed, save_plot_results, set_up_gflownet, get_exact_u_and_P_T
from src.advanced_hypergrid import AdvancedHyperGrid


def evaluate(env, gflownet):
    with torch.no_grad():
        # Compute exact terminating distribution P_T
        _, learned_dist = get_exact_u_and_P_T(env, gflownet.pf)
        true_dist = env.true_dist.cpu()
        l1 = (learned_dist - true_dist).abs().sum().item()

        # Compute Z values
        true_Z = env.get_unormalized_true_dist().sum().item()

        s0 = env.States(torch.zeros(1, env.ndim, dtype=torch.long, device=env.device))
        learned_Z = gflownet.logF(s0).exp().item()

    return {
        'l1': l1,
        'learned_Z': learned_Z,
        'true_Z': true_Z,
    }


def main(args):
    set_seed(0)

    # === Parse w_ratio for multi-objective weighting ===
    if args.w_ratio is not None:
        w_ratio = torch.tensor([float(w) for w in args.w_ratio.split(',')])
        w_ratio = w_ratio.unsqueeze(0)
    else:
        w_ratio = None

    # === Initialize environment ===
    env = AdvancedHyperGrid(
        ndim=args.ndim,
        height=args.height,
        device=args.device,
        custom_dist=args.custom_dist,
        w_ratio=w_ratio,
    )

    # === Initialize preprocessor ===
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # === Build and load GFlowNet ===
    gflownet = set_up_gflownet(
        env, args.hidden_dim, args.n_hidden, preprocessor,
        args.loss, args.subTB_weighting, args.subTB_lambda,
    ).to(args.device)

    gflownet.load_state_dict(torch.load(args.model_saved_path, map_location='cpu', weights_only=True))
    gflownet.eval()

    # === Evaluate ===
    metrics = evaluate(env, gflownet)

    print(f"Evaluation Results:")
    print(f"  L1 Distance: {metrics['l1']:.6f}")
    print(f"  True Z:      {metrics['true_Z']:.4f}")
    print(f"  Learned Z:   {metrics['learned_Z']:.4f}")
    print(f"  Z Diff:      {abs(metrics['true_Z'] - metrics['learned_Z']):.4f}")

    # === Save plot ===
    save_plot_results(
        env=env,
        saved_path=args.result_saved_path,
        pf_estimator=gflownet.pf,
        best_l1=metrics['l1'],
        learned_Z=metrics['learned_Z'],
        true_Z=metrics['true_Z'],
    )
    print(f"Plot saved to: {args.result_saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GFlowNet on Advanced HyperGrid")

    # Model checkpoint
    parser.add_argument("--model_saved_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--result_saved_path", type=str, required=True,
                        help="Path to save evaluation plot")

    # Environment configuration
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--custom_dist", type=str, default=None,
                        help="Reward function: shubert, sphere, etc.")
    parser.add_argument("--w_ratio", type=str, default=None,
                        help="Multi-objective weights, e.g., '0.5,0.5'")

    # Model architecture (must match training)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    args = parser.parse_args()
    main(args)
