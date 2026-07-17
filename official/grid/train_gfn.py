import os
import argparse

import torch
from tqdm import tqdm

from gfn.containers import ReplayBuffer
from gfn.preprocessors import KHotPreprocessor

from src.utils import set_seed, save_plot_results, set_up_gflownet, get_exact_u_and_P_T
from src.advanced_hypergrid import AdvancedHyperGrid


def train(env, optimizer, gflownet, batch_size, n_iterations,
          epsilon, validation_interval, save_dir, save_prefix,
          replay_buffer=None):

    result_saved_path = os.path.join(save_dir, f"{save_prefix}.png")

    val_l1s = []
    val_l1 = None
    learned_Z = None
    best_l1 = float('inf')
    best_learned_Z = None
    prev_saved_path = ""

    true_Z = env.get_unormalized_true_dist().sum().item()

    for it in (pbar := tqdm(range(n_iterations), dynamic_ncols=True)):
        trajectories = gflownet.sample_trajectories(
            env, n=batch_size, save_logprobs=True,
            save_estimator_outputs=False, epsilon=epsilon,
        )
        training_samples = gflownet.to_training_samples(trajectories)

        if replay_buffer is not None:
            with torch.no_grad():
                replay_buffer.add(training_samples)
                training_objects = replay_buffer.sample(n_samples=batch_size)
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        loss = gflownet.loss(
            env, training_objects,
            recalculate_all_logprobs=replay_buffer is not None
        )
        loss.backward()
        optimizer.step()

        if (it + 1) % validation_interval == 0:
            with torch.no_grad():
                _, learned_dist = get_exact_u_and_P_T(env, gflownet.pf)
                true_dist = env.true_dist.cpu()
                val_l1 = (learned_dist - true_dist).abs().sum().item()

                s0 = env.States(torch.zeros(1, env.ndim, dtype=torch.long, device=env.device))
                learned_Z = gflownet.logF(s0).exp().item()

            val_l1s.append(val_l1)

            if val_l1 < best_l1:
                best_l1 = val_l1
                best_learned_Z = learned_Z

            save_plot_results(
                env=env, saved_path=result_saved_path, pf_estimator=gflownet.pf,
                l1_distances=val_l1s, validation_interval=validation_interval,
                best_l1=best_l1, learned_Z=best_learned_Z, true_Z=true_Z,
            )

            if val_l1 <= best_l1:
                if os.path.exists(prev_saved_path):
                    os.remove(prev_saved_path)
                new_path = os.path.join(save_dir, f"{save_prefix}_{it+1}steps.pt")
                torch.save(gflownet.state_dict(), new_path)
                prev_saved_path = new_path

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "l1": f"{val_l1:.6f}" if val_l1 is not None else None,
            "best": f"{best_l1:.6f}" if best_l1 < float('inf') else None,
        })


def main(args):
    set_seed(args.seed)

    if args.w_ratio is not None:
        w_ratio = torch.tensor([float(w) for w in args.w_ratio.split(',')])
        w_ratio = w_ratio.unsqueeze(0)
    else:
        w_ratio = None

    env = AdvancedHyperGrid(
        ndim=args.ndim,
        height=args.height,
        device=args.device,
        custom_dist=args.custom_dist,
        w_ratio=w_ratio,
        beta=args.beta,
    )

    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    gflownet = set_up_gflownet(
        env, args.hidden_dim, args.n_hidden, preprocessor,
        args.loss, args.subTB_weighting, args.subTB_lambda,
        logF_n_hidden=args.logF_n_hidden,
    ).to(args.device)

    replay_buffer = None
    if args.replay_buffer_size > 0:
        replay_buffer = ReplayBuffer(
            env,
            capacity=args.replay_buffer_size,
            prioritized_capacity=False,
        )

    optimizer = torch.optim.Adam(gflownet.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    parts = [args.custom_dist]
    if args.w_ratio is not None:
        parts.append(f"w{'_'.join(args.w_ratio.split(','))}")
    if args.beta != 1.0:
        parts.append(f"beta{args.beta}")
    parts.append(f"seed{args.seed}")
    save_prefix = "_".join(parts)

    train(
        env, optimizer, gflownet, args.batch_size, args.n_iterations,
        args.epsilon, args.validation_interval, args.save_dir, save_prefix,
        replay_buffer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GFlowNet on Advanced HyperGrid")

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Exploration rate for epsilon-greedy sampling")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_iterations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--logF_n_hidden", type=int, default=1,
                        help="Number of hidden layers for logF estimator (default: 1).")
    parser.add_argument("--loss", type=str, choices=["SubTB"], default="SubTB")
    parser.add_argument("--subTB_weighting", type=str, default="geometric_within")
    parser.add_argument("--subTB_lambda", type=float, default=2.0)

    # Validation
    parser.add_argument("--validation_interval", type=int, default=100)

    # Save
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoints and plots")

    # Environment configuration
    parser.add_argument("--custom_dist", type=str, required=True,
                        help="Reward function: shubert, sphere, etc.")
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--w_ratio", type=str, default=None,
                        help="Multi-objective weights, e.g., '0.5,0.5'")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Reward sharpening exponent. >1 sharper, <1 smoother")

    # Replay buffer
    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                        help="Set to 0 to disable replay buffer")

    args = parser.parse_args()
    main(args)
