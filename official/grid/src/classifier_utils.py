"""Shared utilities for classifier training and evaluation on HyperGrid.

Contains common functions used across 2-way and 3-way classifier scripts:
- compute_exact_logp: Exact log-probability computation via BFS
- make_mlp: Simple MLP factory
- toin: Integer coordinates to one-hot encoding
"""

import torch
import torch.nn as nn


def make_mlp(layer_sizes, act=nn.LeakyReLU(), tail=()):
    """Creates a fully-connected network with activation between layers."""
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(layer_sizes) - 2:  # No activation after last layer
            layers.append(act)
    return nn.Sequential(*layers, *tail)


def toin(z, height):
    """Convert integer coordinates to one-hot encoding.
    [batch_size, ndim] -> [batch_size, ndim * height]
    """
    return nn.functional.one_hot(z, height).view(z.shape[0], -1).float()


def compute_exact_logp(fwd_logits_fn, height, ndim, device):
    """Compute exact log probability distribution using forward pass.

    Uses BFS in topological order to compute the unnormalized state-visit
    probability u(s) for all states, then computes P_T(x) = u(x) * P_F(stop|x).

    Args:
        fwd_logits_fn: Function mapping [batch, ndim] integer coords to [batch, n_actions] logits
        height: Grid height per dimension
        ndim: Number of dimensions
        device: Torch device for the result

    Returns:
        Log probability tensor of shape [height^ndim] on the given device
    """
    u = torch.ones(height ** ndim, dtype=torch.float32)

    all_states = torch.zeros((height ** ndim, ndim), dtype=torch.long)
    for i in range(height):
        for j in range(height):
            idx = i * height + j
            all_states[idx, 0] = i
            all_states[idx, 1] = j

    for total in range(2 * (height - 1) + 1):
        for i in range(max(0, total - height + 1), min(total + 1, height)):
            j = total - i
            if j < 0 or j >= height:
                continue

            idx = i * height + j
            if i == 0 and j == 0:
                u[idx] = 1.0
                continue

            z = all_states[idx:idx+1]
            logits = fwd_logits_fn(z).cpu()
            probs = torch.softmax(logits, dim=1)[0]

            u_val = 0.0
            if i > 0:
                parent_idx = (i - 1) * height + j
                parent_z = all_states[parent_idx:parent_idx+1]
                parent_logits = fwd_logits_fn(parent_z).cpu()
                parent_probs = torch.softmax(parent_logits, dim=1)[0]
                u_val += u[parent_idx] * parent_probs[0]
            if j > 0:
                parent_idx = i * height + (j - 1)
                parent_z = all_states[parent_idx:parent_idx+1]
                parent_logits = fwd_logits_fn(parent_z).cpu()
                parent_probs = torch.softmax(parent_logits, dim=1)[0]
                u_val += u[parent_idx] * parent_probs[1]

            u[idx] = u_val

    all_logits = fwd_logits_fn(all_states).cpu()
    all_probs = torch.softmax(all_logits, dim=1)
    P_T = u * all_probs[:, -1]

    return torch.log(P_T + 1e-40).to(device)
