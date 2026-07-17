"""Build representative preference vectors on the (dim-1)-simplex.

`simplex_representatives(dim, n)` returns `n` evenly-spread points on the simplex; it is
the public entry point used by `eval_loops.build_grid_cond_vectors` when `pref_mode="simplex"`.

Internally:
  1. `simplex_lattice(dim, m)` enumerates all lattice points with coordinates in 1/m-steps
     (stars-and-bars; total count is `C(m+dim-1, dim-1)`).
  2. We pick the smallest `m` whose lattice has ≥ `n` points.
  3. `_fps` does farthest-point sampling on that lattice to thin it down to exactly `n`
     well-spread representatives.
"""
import itertools
from math import comb

import torch


def simplex_lattice(dim: int, m: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    All points on the (dim-1)-simplex with coordinates multiples of 1/m.
    Returns tensor shape [C, dim], where C = comb(m+dim-1, dim-1).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    points = []
    # stars-and-bars: choose (dim-1) cuts among (m+dim-1) slots
    for cuts in itertools.combinations(range(m + dim - 1), dim - 1):
        prev = -1
        ks = []
        for c in cuts:
            ks.append(c - prev - 1)
            prev = c
        ks.append(m + dim - 1 - prev - 1)
        points.append(ks)

    pts = torch.tensor(points, device=device, dtype=dtype) / float(m)
    return pts  # [C, dim], each row sums to 1

# ---------- farthest-point sampling for coverage ----------
def _fps(X: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """
    Greedy maximin selection of k indices from rows of X (shape [N, d]).
    """
    g = torch.Generator(device=X.device).manual_seed(seed)
    start = int(torch.randint(X.shape[0], (1,), generator=g, device=X.device))
    sel = [start]
    d2 = ((X - X[start])**2).sum(dim=1)

    for _ in range(1, k):
        nxt = int(torch.argmax(d2))
        sel.append(nxt)
        d2 = torch.minimum(d2, ((X - X[nxt])**2).sum(dim=1))
    return torch.tensor(sel, device=X.device, dtype=torch.long)

def simplex_representatives(dim: int, n: int, device=None, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    """
    Exactly n representative simplex points via a lattice + farthest-point sampling.
    Picks the smallest m with enough lattice points, then selects n.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    m = 0
    while comb(m + dim - 1, dim - 1) < n:
        m += 1
    lattice = simplex_lattice(dim, m, device=device, dtype=dtype)
    if lattice.shape[0] == n:
        return lattice
    idx = _fps(lattice, n, seed=seed)
    return lattice[idx]  # [n, dim]
