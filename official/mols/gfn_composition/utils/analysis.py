"""DataFrame helpers for eval post-processing and downstream analysis.

  - `add_scores_and_filter` is called by `postprocess_db` after every eval to backfill
    per-objective columns in the generated `.db` files.
  - `compute_bin_statistics` / `compute_bin_statistics_df` partition samples by per-
    objective threshold and report L/H bin counts (useful for analyzing how well each
    method covers the high-reward corner of multi-objective space).
  - `topk_avg_tanimoto_df` measures chemical diversity within the top-K rewards.
"""
import itertools

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs


def compute_bin_statistics(rewards, reward_names, thresholds):
    """Bucket each row of `rewards` into 2^k bins by per-objective threshold; return
    bin → percent-of-total. `bin_label` is a "L"/"H" string of length k (e.g. "HLH")."""
    n_rewards = len(reward_names)
    t_arr = [thresholds[name] for name in reward_names]
    combinations = list(itertools.product([0, 1], repeat=n_rewards))
    results = {}
    total_samples = len(rewards)
    for combo in combinations:
        mask = np.ones(rewards.shape[0], dtype=bool)
        for i, t in enumerate(t_arr):
            if combo[i] == 0:
                mask &= rewards[:, i] < t
            else:
                mask &= rewards[:, i] >= t
        count = np.sum(mask)
        bin_label = "".join(["H" if c == 1 else "L" for c in combo])
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        results[bin_label] = pct
    return results


def compute_bin_statistics_df(df, reward_names, thresholds, return_counts=False):
    """DataFrame-based variant of `compute_bin_statistics` with human-readable bin labels
    ("low gap, high sa, ...") and optional raw-count instead of percentage."""
    n_rewards = len(reward_names)
    assert n_rewards in [2, 3], "Only 2 or 3 rewards are supported"
    r_mat = df[reward_names].values
    t_arr = [thresholds[name] for name in reward_names]
    combinations = list(itertools.product([0, 1], repeat=n_rewards))
    results = []
    total_samples = len(df)
    for combo in combinations:
        mask = np.ones(r_mat.shape[0], dtype=bool)
        for i, t in enumerate(t_arr):
            if combo[i] == 0:
                mask &= r_mat[:, i] < t
            else:
                mask &= r_mat[:, i] >= t
        count = np.sum(mask)
        labels = [f"{'high' if combo[i] == 1 else 'low'} {n}" for i, n in enumerate(reward_names)]
        bin_label = ", ".join(labels)
        if return_counts:
            results.append({"bin": bin_label, "count": count})
        else:
            pct = (count / total_samples * 100) if total_samples > 0 else 0
            results.append({"bin": bin_label, "percentage": pct})
    return pd.DataFrame(results)


def topk_avg_tanimoto_df(df: pd.DataFrame, k: int = 10, reward_col: str = "r"):
    """Top-K rows by reward, return (avg_sim, diversity, indices)."""
    topk = df.nlargest(k, reward_col)
    fps = []
    sel_df_idx = []
    for i, smi in topk["smi"].items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fps.append(Chem.RDKFingerprint(m))
        sel_df_idx.append(i)

    if not fps:
        return np.nan, np.nan, []

    n = len(fps)
    if n == 1:
        return 0.0, 1.0, sel_df_idx
    sims = []
    for a in range(n):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[a], fps[a + 1:]))
    avg_sim = float(np.mean(sims)) if sims else 0.0
    return avg_sim, 1.0 - avg_sim, sel_df_idx


def add_scores_and_filter(df: pd.DataFrame, trainer) -> pd.DataFrame:
    """For each SMILES in `df`, recompute every objective of `trainer.task` and add the
    scores as named columns; drop rows whose SMILES doesn't parse. Used by
    `postprocess_db` to backfill `.db` files with per-objective columns."""
    df = df.copy().reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in df["smi"]]
    scores, is_valid = trainer.task.compute_obj_properties(mols)
    if hasattr(is_valid, "detach"):
        is_valid = is_valid.detach().cpu().numpy()
    is_valid = np.asarray(is_valid, dtype=bool)
    scores = scores.detach().cpu().numpy()

    df_valid = df[is_valid].copy().reset_index(drop=True)
    assert scores.shape[0] == len(df_valid), (
        f"Mismatch: got {scores.shape[0]} scores for {len(df_valid)} valid molecules"
    )
    obj_names = trainer.task.objectives
    for i, name in enumerate(obj_names):
        df_valid[name] = scores[:, i]
    return df_valid


