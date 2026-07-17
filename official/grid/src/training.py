import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.simplex import simplex_list
from src.utils import get_exact_u_and_P_T


def validate_conditional(env, pf_estimator, conditioning):
    """Compute exact L1 distance for a single conditioning vector."""
    with torch.no_grad():
        _, learned_dist = get_exact_u_and_P_T(env, pf_estimator, cond=conditioning)
        true_dist = env.true_dist(conditioning).cpu()
        l1_dist = (learned_dist - true_dist).abs().sum().item()
    return l1_dist


def save_l1_progress_plot(l1_distances, validation_interval, best_l1, saved_path):
    """Save L1 progress plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    steps = [(i + 1) * validation_interval for i in range(len(l1_distances))]
    ax.plot(steps, l1_distances, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean L1 Error")
    ax.set_title(f"L1 Error Evolution (Best: {best_l1:.6f})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(saved_path, dpi=150, bbox_inches="tight")
    plt.close()


def train_conditional(env, optimizer, gflownet, batch_size, n_iterations, epsilon,
                      validation_interval, n_reward_fns, n_simplex_for_val,
                      save_dir, save_prefix, replay_buffer=None):
    """Train conditional GFlowNet on an AdvancedConditionalHyperGrid.

    Sampling flow (torchgfn 2.4.1):
        - `gflownet.sample_trajectories(env, n=batch_size)` triggers
          `env.sample_conditions((batch_size,))` internally and attaches the
          sampled conditions to `states.conditions`.
        - `env.reward(states)` reads from `states.conditions`, so log_rewards
          are computed per-trajectory with the correct condition.
    """
    result_saved_path = os.path.join(save_dir, f"{save_prefix}.png")

    val_mean_l1s = []
    val_mean_l1 = None
    best_val_mean_l1 = float('inf')
    prev_saved_path = ""

    simplex = simplex_list(n_reward_fns, n_simplex_for_val, env.device)
    is_on_policy = (epsilon == 0.0) and (replay_buffer is None)

    for it in (pbar := tqdm(range(n_iterations), dynamic_ncols=True)):
        # env.sample_conditions is called inside sample_trajectories
        trajectories = gflownet.sample_trajectories(
            env, n=batch_size,
            save_logprobs=is_on_policy,
            save_estimator_outputs=not is_on_policy,
            epsilon=epsilon,
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
            recalculate_all_logprobs=not is_on_policy,
        )
        loss.backward()
        optimizer.step()

        if (it + 1) % validation_interval == 0:
            l1_errors = []
            for cond in simplex:
                l1 = validate_conditional(env, gflownet.pf, cond)
                l1_errors.append(l1)

            val_mean_l1 = sum(l1_errors) / len(l1_errors)
            val_mean_l1s.append(val_mean_l1)

            if val_mean_l1 <= best_val_mean_l1:
                best_val_mean_l1 = val_mean_l1
                if os.path.exists(prev_saved_path):
                    os.remove(prev_saved_path)
                new_path = os.path.join(save_dir, f"{save_prefix}_{it+1}steps.pt")
                torch.save(gflownet.state_dict(), new_path)
                prev_saved_path = new_path

            save_l1_progress_plot(
                l1_distances=val_mean_l1s,
                validation_interval=validation_interval,
                best_l1=best_val_mean_l1,
                saved_path=result_saved_path,
            )

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "l1": f"{val_mean_l1:.6f}" if val_mean_l1 is not None else None,
            "best": f"{best_val_mean_l1:.6f}" if best_val_mean_l1 < float('inf') else None,
        })


def evaluate_conditional(env, gflownet, n_reward_fns, n_simplex_for_val,
                         result_saved_path=None):
    """Evaluate conditional GFlowNet. Returns mean_l1 and l1_errors.

    If `result_saved_path` is given, saves a visualization with 5 representative
    simplex points (True vs Learned) and an L1 histogram.
    """
    simplex = simplex_list(n_reward_fns, n_simplex_for_val, env.device)

    print(f"Evaluating on {len(simplex)} simplex points...")

    n_vis = min(5, len(simplex))
    vis_indices = np.linspace(0, len(simplex) - 1, n_vis, dtype=int).tolist()

    l1_errors = []
    vis_data = []

    with torch.no_grad():
        for i, cond in enumerate(tqdm(simplex)):
            true_dist = env.true_dist(cond).cpu()
            _, learned_dist = get_exact_u_and_P_T(env, gflownet.pf, cond)

            l1 = (learned_dist - true_dist).abs().sum().item()
            l1_errors.append(l1)

            if i in vis_indices:
                vis_data.append((cond.squeeze(0).cpu(), true_dist, learned_dist, l1))

    mean_l1 = sum(l1_errors) / len(l1_errors)

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

            axes[0, col].imshow(td_img, cmap="Blues", origin="upper",
                                vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[0, col].set_title(f"True w=[{w_str}]", fontsize=9)
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])

            axes[1, col].imshow(ld_img, cmap="Blues", origin="upper",
                                vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[1, col].set_title(f"Learned L1={l1:.4f}", fontsize=9)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])

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
