"""Load pre-trained single-objective base GFN checkpoints as `ingredient_model_trainers`
for mixing eval. See `eval_scalarization.py` / `eval_logical_operators.py` for the call site.
"""
import os

import torch

from gfn_composition.tasks.qm9 import BaseQM9Trainer
from gfn_composition.tasks.seh_frag import BaseSEHFragTrainer
from gfn_composition.utils.eval_config import (
    frag_config_for_loading,
    qm9_config_for_loading,
)


def load_ingredient_model_trainers(ckpt_paths, dist_params=(96.0,), device="cpu",
                                   score_names=("seh", "qed"), task="frag",
                                   seed=0, max_nodes=9):
    """Build one base-GFN Trainer per (ckpt, score_name) pair and load weights.

    Returns a list of trainers in the same order as `ckpt_paths` / `score_names`. The
    mixture samplers downstream rely on this ordering matching the preferences vector.
    """
    trainers = []
    for i, ckpt_path in enumerate(ckpt_paths):
        # Unique throwaway dir per ingredient — PID suffix avoids race across concurrent jobs.
        dir_name = f"./logs/tmp_ingredient_{i}_{score_names[i]}_{os.getpid()}"
        if task == "frag":
            cfg = frag_config_for_loading(dir_name=dir_name, obj_param=[score_names[i]],
                                           dist_params=list(dist_params), device=device, seed=seed,
                                           max_nodes=max_nodes)
            cfg.overwrite_existing_exp = True
            t = BaseSEHFragTrainer(cfg, print_config=False)
        elif task == "qm9":
            cfg = qm9_config_for_loading(dir_name=dir_name, obj_param=[score_names[i]],
                                          dist_params=list(dist_params), device=device, seed=seed,
                                          max_nodes=max_nodes)
            cfg.overwrite_existing_exp = True
            t = BaseQM9Trainer(cfg, print_config=False)
        else:
            raise ValueError(f"Unknown task: {task}")

        state_dict = torch.load(ckpt_path)
        t.model.load_state_dict(state_dict["models_state_dict"][0], strict=True)
        t.model.to(device)
        trainers.append(t)
        print(f">>>>> Load Done {ckpt_path}")
    return trainers
