import os
import argparse

import torch

from gfn.containers import ReplayBuffer
from gfn.preprocessors import KHotPreprocessor

from src.utils import set_seed, set_up_hn_gflownet
from src.advanced_hypergrid import AdvancedConditionalHyperGrid
from src.training import train_conditional


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

    predictor_layers = [int(x) for x in args.predictor_layers.split(",") if x.strip()]

    gflownet = set_up_hn_gflownet(
        env=env,
        preprocessor=preprocessor,
        n_reward_fns=args.n_reward_fns,
        ray_hidden_dim=args.ray_hidden_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        subTB_weighting=args.subTB_weighting,
        subTB_lambda=args.subTB_lambda,
        predictor_layers=predictor_layers,
        logit_clipping=args.logit_clipping,
    ).to(args.device)

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    replay_buffer = None
    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            env,
            capacity=args.replay_buffer_size,
            prioritized_capacity=False,
        )

    os.makedirs(args.save_dir, exist_ok=True)
    save_prefix = f"hngfn_{args.custom_dist}_seed{args.seed}"

    train_conditional(
        env, optimizer, gflownet, args.batch_size, args.n_iterations,
        args.epsilon, args.validation_interval,
        args.n_reward_fns, args.n_simplex_for_val,
        args.save_dir, save_prefix,
        replay_buffer=replay_buffer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HN-GFN on AdvancedConditionalHyperGrid")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)

    # HN-GFN architecture
    parser.add_argument("--ray_hidden_dim", type=int, default=32,
                        help="Hidden dimension for HyperNetwork's ray MLP")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension for target policy network")
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--predictor_layers", type=str, default="",
                        help="Hidden dims for generated predictor head, e.g., '64,64'. Empty for single linear layer.")
    parser.add_argument("--logit_clipping", type=float, default=0.0)
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--n_simplex_for_val", type=int, default=128)

    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoints and plots")

    parser.add_argument("--custom_dist", type=str, required=True,
                        help="Reward function: mix2, mix3, etc.")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--n_reward_fns", type=int, default=2)

    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                        help="Set to 0 to disable replay buffer")

    args = parser.parse_args()
    main(args)
