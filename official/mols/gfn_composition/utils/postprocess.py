"""Post-process generated `.db` files: compute every task objective per molecule and
add named columns. Used after `gfn` single-objective generation (where the .db only
has `fr_0` from one objective) and after logical-operator mixing (where the sampler
doesn't track preference-keyed scores).

Also exposes `write_summary_txt`, a small helper that writes a human-readable
`summary.txt` next to the `.db` files.
"""
import glob
import os
import sqlite3
from typing import List

import numpy as np
import pandas as pd

from gfn_composition.tasks.qm9_moo import MultiObjectiveQM9Trainer
from gfn_composition.tasks.seh_frag_moo import MultiObjectiveSEHFragTrainer
from gfn_composition.utils.analysis import (
    add_scores_and_filter,
    compute_bin_statistics_df,
    topk_avg_tanimoto_df,
)
from gfn_composition.utils.eval_config import (
    frag_moo_config_for_loading,
    qm9_moo_config_for_loading,
)


_SCORE_NAMES = {
    "frag": ["seh", "sa", "qed"],
    "qm9":  ["gap", "sa", "qed"],
}


def postprocess_db(dir_name: str, task: str, dist_params):
    """Write per-objective columns into every `<dir_name>/final/*.db` file."""
    score_names: List[str] = _SCORE_NAMES[task]
    dist_params = list(dist_params) if isinstance(dist_params, (list, tuple)) else [dist_params]

    if task == "frag":
        cfg = frag_moo_config_for_loading(
            dir_name="./logs/tmp_postprocess",
            obj_param=score_names, dist_params=dist_params,
        )
        cfg.overwrite_existing_exp = True
        trainer = MultiObjectiveSEHFragTrainer(cfg, print_config=False)
    else:
        cfg = qm9_moo_config_for_loading(
            dir_name="./logs/tmp_postprocess",
            obj_param=score_names, dist_params=dist_params,
        )
        cfg.overwrite_existing_exp = True
        trainer = MultiObjectiveQM9Trainer(cfg, print_config=False)

    final_dir = os.path.join(dir_name, "final")
    db_files = sorted(f for f in os.listdir(final_dir) if f.endswith(".db"))

    for db_file in db_files:
        db_path = os.path.join(final_dir, db_file)
        df = pd.read_sql_query("SELECT * FROM results", sqlite3.connect(db_path))
        if len(df) == 0:
            continue
        df_scored = add_scores_and_filter(df, trainer)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for col in score_names:
            try:
                cur.execute(f"ALTER TABLE results ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass
        for _, row in df_scored.iterrows():
            values = [float(row[col]) for col in score_names]
            set_clause = ", ".join(f"{col} = ?" for col in score_names)
            cur.execute(f"UPDATE results SET {set_clause} WHERE smi = ?", values + [row["smi"]])
        conn.commit()
        conn.close()

    print(f"Postprocess done: added columns {score_names} to {len(db_files)} db files")


def write_summary_txt(dir_name: str, task: str, obj_param, dist_params,
                       thresholds: dict):
    """Concatenate every `<dir_name>/final/*.db`, compute basic stats + an L/H bin
    breakdown across the three task objectives, and write `<dir_name>/summary.txt`.

    `thresholds` is a `{objective_name: float}` dict (callers should define their
    own — for `frag` the names are `seh / sa / qed`, for `qm9` `gap / sa / qed`).

    Reports: total / valid / unique counts, mean `r`, and a 2^k bin table where
    each row is "low/high <obj> × k" with the row's percentage of valid samples.
    """
    score_names: List[str] = _SCORE_NAMES[task]

    final_dir = os.path.join(dir_name, "final")
    db_files = sorted(glob.glob(os.path.join(final_dir, "*.db")))

    frames = []
    for path in db_files:
        try:
            frames.append(pd.read_sql_query("SELECT * FROM results", sqlite3.connect(path)))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    total = len(df)
    valid_mask = df["smi"].notna() & (df["smi"] != "") if total else pd.Series([], dtype=bool)
    valid = int(valid_mask.sum())
    df_valid = df[valid_mask].copy() if total else df
    unique = int(df_valid["smi"].nunique()) if valid else 0
    mean_r = float(df_valid["r"].mean()) if valid else float("nan")

    bin_df = (
        compute_bin_statistics_df(df_valid, score_names, thresholds)
        if valid and all(n in df_valid.columns for n in score_names)
        else None
    )

    th_str = ", ".join(f"{n}={thresholds[n]}" for n in score_names)
    out_path = os.path.join(dir_name, "summary.txt")
    with open(out_path, "w") as f:
        f.write(f"Task         : {task}\n")
        f.write(f"Objective(s) : {list(obj_param)}\n")
        f.write(f"dist_params  : {list(dist_params) if isinstance(dist_params, (list, tuple)) else [dist_params]}\n")
        f.write(f"DB files     : {len(db_files)}\n")
        f.write(f"Total        : {total}\n")
        f.write(f"Valid        : {valid}\n")
        f.write(f"Unique       : {unique}\n")
        f.write(f"Mean reward  : {mean_r:.4f}\n")
        f.write(f"\n=== Bin statistics (% of valid) ===\n")
        f.write(f"Thresholds   : {th_str}\n")
        if bin_df is None:
            f.write("  (skipped: missing per-objective columns; run postprocess_db first)\n")
        else:
            for _, row in bin_df.iterrows():
                f.write(f"  {row['bin']:48s} : {row['percentage']:7.3f}%\n")
    print(f"Summary written to {out_path}")


def write_logical_summary_txt(dir_name: str, task: str, obj_param, dist_params,
                               thresholds: dict):
    """Logical-operator-eval summary (HM / contrast) written to `<dir_name>/summary.txt`.
    Reports Trials / Valid / Unique + an L/H bin breakdown across the three task
    objectives (same bin format as `write_summary_txt`, no preference axis).
    """
    score_names: List[str] = _SCORE_NAMES[task]

    final_dir = os.path.join(dir_name, "final")
    db_files = sorted(glob.glob(os.path.join(final_dir, "*.db")))

    frames = []
    for path in db_files:
        try:
            frames.append(pd.read_sql_query("SELECT * FROM results", sqlite3.connect(path)))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    trials = len(df)
    valid_mask = df["smi"].notna() & (df["smi"] != "") if trials else pd.Series([], dtype=bool)
    valid = int(valid_mask.sum())
    df_valid = df[valid_mask].copy() if trials else df
    unique = int(df_valid["smi"].nunique()) if valid else 0

    bin_df = (
        compute_bin_statistics_df(df_valid, score_names, thresholds)
        if valid and all(n in df_valid.columns for n in score_names)
        else None
    )

    th_str = ", ".join(f"{n}={thresholds[n]}" for n in score_names)
    out_path = os.path.join(dir_name, "summary.txt")
    with open(out_path, "w") as f:
        f.write(f"Task         : {task}\n")
        f.write(f"Objective(s) : {list(obj_param)}\n")
        f.write(f"dist_params  : {list(dist_params) if isinstance(dist_params, (list, tuple)) else [dist_params]}\n")
        f.write(f"DB files     : {len(db_files)}\n")
        f.write(f"Trials       : {trials}\n")
        f.write(f"Valid        : {valid}\n")
        f.write(f"Unique       : {unique}\n")
        f.write(f"\n=== Bin statistics (% of valid) ===\n")
        f.write(f"Thresholds   : {th_str}\n")
        if bin_df is None:
            f.write("  (skipped: missing per-objective columns; run postprocess_db first)\n")
        else:
            for _, row in bin_df.iterrows():
                f.write(f"  {row['bin']:48s} : {row['percentage']:7.3f}%\n")
    print(f"Summary written to {out_path}")


def write_mixing_summary_txt(dir_name: str, task: str, obj_param, dist_params, top_k: int = 10):
    """Mixing-eval summary written to `<dir_name>/summary.txt`. Reports:
        Trials / Valid / Unique / #Prefs / Reward(topK) / Diversity(topK)

    `Reward (topK)` and `Diversity (topK)` are *per-preference* averages: for each
    unique preference vector we take the top-K molecules by `r`, compute the mean
    reward and the top-K Tanimoto diversity, and then average those per-pref values.
    Used by `eval_scalarization.py` (and suitable for any mixing eval whose `.db`
    rows carry `r` + `smi` + zero-or-more `pref_*` columns).
    """
    final_dir = os.path.join(dir_name, "final")
    db_files = sorted(glob.glob(os.path.join(final_dir, "*.db")))

    frames = []
    for path in db_files:
        try:
            frames.append(pd.read_sql_query("SELECT * FROM results", sqlite3.connect(path)))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    trials = len(df)
    valid_mask = df["smi"].notna() & (df["smi"] != "") if trials else pd.Series([], dtype=bool)
    valid = int(valid_mask.sum())
    df_valid = df[valid_mask].copy() if trials else df
    unique = int(df_valid["smi"].nunique()) if valid else 0

    pref_cols = [c for c in df_valid.columns if c.startswith("pref_")]
    unique_prefs = (
        df_valid[pref_cols].drop_duplicates().reset_index(drop=True)
        if valid and pref_cols
        else pd.DataFrame()
    )
    n_prefs = len(unique_prefs)

    # Per-pref top-K: take the top_k rows by `r` within each preference group,
    # then mean across prefs. Same for the Tanimoto top-K diversity.
    reward_per_pref, diversity_per_pref = [], []
    for _, pref_row in unique_prefs.iterrows():
        mask = (df_valid[pref_cols] == pref_row.values).all(axis=1)
        df_p = df_valid[mask]
        if len(df_p) < top_k:
            continue
        topk = df_p.nlargest(top_k, "r")
        reward_per_pref.append(float(topk["r"].mean()))
        _avg_sim, div, _idx = topk_avg_tanimoto_df(df_p, k=top_k, reward_col="r")
        diversity_per_pref.append(float(div))

    reward_topk = float(np.mean(reward_per_pref)) if reward_per_pref else float("nan")
    diversity_topk = float(np.mean(diversity_per_pref)) if diversity_per_pref else float("nan")

    out_path = os.path.join(dir_name, "summary.txt")
    with open(out_path, "w") as f:
        f.write(f"Task                       : {task}\n")
        f.write(f"Objective(s)               : {list(obj_param)}\n")
        f.write(f"dist_params                : {list(dist_params) if isinstance(dist_params, (list, tuple)) else [dist_params]}\n")
        f.write(f"DB files                   : {len(db_files)}\n")
        f.write(f"Trials                     : {trials}\n")
        f.write(f"Valid                      : {valid}\n")
        f.write(f"Unique                     : {unique}\n")
        f.write(f"#Prefs                     : {n_prefs}\n")
        f.write(f"Reward (mean per-pref top{top_k})    : {reward_topk:.4f}\n")
        f.write(f"Diversity (mean per-pref top{top_k}) : {diversity_topk:.4f}\n")
    print(f"Summary written to {out_path}")
