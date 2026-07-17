"""Eval-time grid sampling using upstream's `RepeatedCondInfoDataset` pattern.

For multi-objective eval (mixing) we sweep a deterministic grid
of preferences and sample many molecules per preference. This is the upstream-intended
flow (`do_conditionals_dataset_in_order` over a `RepeatedCondInfoDataset`), distinct from
`do_sample_model_n_times` which would re-sample preferences each iter.

Two modes:
  - "simplex": N evenly distributed grid points on the simplex (Pareto sweep).
  - "fixed":   1 user-supplied preference vector (deep sample at one point).

Total samples = len(grid) * n_repeat. Batch size is independent of grid size, so any
batch_size can be picked for GPU efficiency.
"""
import glob
import pathlib
import sqlite3

import numpy as np

from gflownet.data.data_source import DataSource
from gflownet.tasks.seh_frag_moo import RepeatedCondInfoDataset

from gfn_composition.data.mixture_data_source import MixtureDataSource
from gfn_composition.utils.simplex import simplex_representatives
from gfn_composition.utils.sqlite_log import NamedColumnSQLiteLogHook


def _count_valid_in_db(final_dir):
    """Count valid mols (smi != '') across all generated_objs_*.db files."""
    total = 0
    for db in glob.glob(f"{final_dir}/generated_objs_*.db"):
        try:
            conn = sqlite3.connect(db)
            total += conn.execute(
                "SELECT COUNT(*) FROM results WHERE smi != ''"
            ).fetchone()[0]
            conn.close()
        except Exception:
            pass
    return total


def build_grid_cond_vectors(num_objectives, pref_mode, *, n_pref=10, fixed_pref=None, focus_dim=0):
    """Build the (N, num_objectives + focus_dim) ndarray fed to `RepeatedCondInfoDataset`.

    Each row is a "steer_info" entry that `task.encode_conditional_information(...)` accepts.
    """
    if pref_mode == "simplex":
        prefs = simplex_representatives(num_objectives, n_pref, device="cpu").cpu().numpy()
    elif pref_mode == "fixed":
        if fixed_pref is None:
            raise ValueError("fixed_pref required for pref_mode='fixed'")
        if len(fixed_pref) != num_objectives:
            raise ValueError(f"fixed_pref len {len(fixed_pref)} != num_objectives {num_objectives}")
        prefs = np.array([fixed_pref], dtype=np.float64)
    else:
        raise ValueError(f"unknown pref_mode {pref_mode!r}")

    if focus_dim > 0:
        prefs = np.concatenate([prefs, np.zeros((prefs.shape[0], focus_dim))], axis=1)
    return prefs


def run_grid_eval(trainer, cond_vectors, n_repeat, batch_size, output_dir,
                  mixture_mode=False, target_valid=None, max_retry_factor=2.0):
    """Iterate (cond_vectors × n_repeat) in batches of `batch_size`, sample, log to `output_dir`.

    Layout of `RepeatedCondInfoDataset(cond_vectors, repeat=K)` is `[c0]*K, [c1]*K, ...`,
    so consecutive batches walk through the prefs in order (one pref dominates each batch
    when batch_size <= K).

    `mixture_mode=True` uses `MixtureDataSource` which forwards the full cond_info dict
    (including raw `preferences`) to the algo, so a `MixtureGraphSampler*` can read
    `total_cond_info["preferences"]`.

    If `target_valid` is given, after the initial pass the loop keeps sampling extra
    cond-vector rounds until at least `target_valid` valid mols are in the db, capped
    at `max_retry_factor` × initial-budget. Used by logical-operator modes (HM / CT)
    where invalid-mol rates can be high and we want a comparable count of valid samples.
    """
    test_data = RepeatedCondInfoDataset(cond_vectors, repeat=n_repeat)
    n_total = len(test_data)
    n_batches = (n_total + batch_size - 1) // batch_size

    ds_cls = MixtureDataSource if mixture_mode else DataSource

    def _make_src_with_data(test_data_):
        src = ds_cls(trainer.cfg, trainer.ctx, trainer.algo, trainer.task, is_algo_eval=True)
        src.do_conditionals_dataset_in_order(test_data_, batch_size, trainer.model)
        src.add_sampling_hook(NamedColumnSQLiteLogHook(
            str(output_dir), trainer.ctx, obj_names=trainer.task.objectives,
        ))
        return src

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = _make_src_with_data(test_data)
    dl = trainer._make_data_loader(src)
    for i, _batch in enumerate(dl, start=1):
        print(f"  Grid eval batch {i}/{n_batches}", flush=True)
    print(f"Grid eval pass 1 done: {len(cond_vectors)} prefs × {n_repeat} repeats = {n_total} samples")

    if target_valid is None:
        return

    # ---- retry: keep sampling extra cond-vector rounds until enough valid mols ----
    max_attempts = int(n_total * max_retry_factor)
    attempts_so_far = n_total
    valid_count = _count_valid_in_db(str(output_dir))
    print(f"  initial valid={valid_count}/{target_valid} (attempts={attempts_so_far})")

    # Each retry round samples at least `batch_size` more (one batch), proportionally per pref.
    extra_repeat = max(1, batch_size // max(1, len(cond_vectors)))

    while valid_count < target_valid and attempts_so_far < max_attempts:
        extra_data = RepeatedCondInfoDataset(cond_vectors, repeat=extra_repeat)
        src_extra = _make_src_with_data(extra_data)
        dl_extra = trainer._make_data_loader(src_extra)
        for _batch in dl_extra:
            pass
        attempts_so_far += len(extra_data)
        valid_count = _count_valid_in_db(str(output_dir))
        print(f"  retry: attempts={attempts_so_far}, valid={valid_count}/{target_valid}")

    if valid_count < target_valid:
        print(f"  ⚠ stopped at attempts={attempts_so_far} (cap={max_attempts}) "
              f"with valid={valid_count} < target={target_valid}")
    print(f"Grid eval done: total attempts={attempts_so_far}, valid={valid_count}")
