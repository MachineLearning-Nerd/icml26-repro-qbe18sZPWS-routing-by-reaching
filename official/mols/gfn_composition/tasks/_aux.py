"""Per-objective reward functions shared across SEH/QM9 tasks.

These compute scalar [0, 1]-ish rewards for individual molecules from RDKit Mol objects.
Used by the multi-objective task classes to compute per-objective rewards for non-proxy
objectives (seh / gap come from learned proxies and are handled separately).
"""
import torch
from rdkit.Chem import QED

from gflownet.utils import sascore


def safe(f, x, default):
    """Call `f(x)` and return `default` if it raises (e.g. RDKit failure on edge-case mols)."""
    try:
        return f(x)
    except Exception:
        return default


def mol2sas(mols, is_valid, default=10):
    """Synthetic Accessibility score in [0, 1] (higher = easier to synthesize).

    Raw SA is in [1, 10] with smaller-is-better; we map via (10 - SA) / 9 to flip the
    direction and rescale to [0, 1].
    """
    sas = torch.tensor([safe(sascore.calculateScore, i, default) for i, v in zip(mols, is_valid) if v])
    return (10 - sas) / 9


def mol2qed(mols, is_valid, default=0):
    """Quantitative Estimation of Drug-likeness in [0, 1] (higher = more drug-like)."""
    return torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v])


# Dispatch table for per-objective reward functions, used by the task wrappers when the
# objective is not the proxy-backed one (`seh` for frag, `gap` for QM9).
AUX_TASKS = {"qed": mol2qed, "sa": mol2sas}
