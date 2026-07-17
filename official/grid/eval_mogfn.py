import argparse

import torch

from gfn.preprocessors import KHotPreprocessor

from src.utils import set_seed, set_up_conditional_gflownet
from src.advanced_hypergrid import AdvancedConditionalHyperGrid
from src.training import evaluate_conditional


def main(args):
    set_seed(args.seed)

    env = AdvancedConditionalHyperGrid(
        ndim=args.ndim,
        height=args.height,
        n_reward_fns=args.n_reward_fns,
        device=args.device,
        custom_dist=args.custom_dist,
    )

    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    gflownet = set_up_conditional_gflownet(
        env, args.hidden_dim, args.n_hidden, preprocessor,
        args.loss, args.n_reward_fns, args.cond_hidden_dim,
        args.subTB_weighting, args.subTB_lambda,
    ).to(args.device)

    gflownet.load_state_dict(torch.load(args.model_saved_path, map_location='cpu', weights_only=True))
    gflownet.eval()

    metrics = evaluate_conditional(
        env, gflownet, args.n_reward_fns, args.n_simplex_for_val,
        result_saved_path=args.result_saved_path,
    )

    print(f"\nEvaluation Results:")
    print(f"  Mean L1 Distance: {metrics['mean_l1']:.6f}")
    print(f"  Min L1:  {min(metrics['l1_errors']):.6f}")
    print(f"  Max L1:  {max(metrics['l1_errors']):.6f}")
    if args.result_saved_path:
        print(f"  Plot saved to: {args.result_saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MO-GFN on Advanced HyperGrid")

    # Model checkpoint
    parser.add_argument("--model_saved_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--result_saved_path", type=str, default=None,
                        help="Path to save evaluation plot (5 representative points + L1 histogram)")

    # Environment configuration
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--custom_dist", type=str, required=True,
                        help="Reward function: mix2, mix3, etc.")
    parser.add_argument("--n_reward_fns", type=int, default=2,
                        help="Number of objectives (reward functions)")

    # Model architecture (must match training)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--cond_hidden_dim", type=int, default=16)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    # Evaluation
    parser.add_argument("--n_simplex_for_val", type=int, default=128,
                        help="Number of evenly spaced simplex points for evaluation")

    args = parser.parse_args()
    main(args)
