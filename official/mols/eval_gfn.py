"""Single-objective base GFN evaluation: generate molecules from one trained model
and write per-objective scores to a SQLite log.
"""
import argparse

import numpy as np
import torch

from gfn_composition.tasks.qm9 import BaseQM9Trainer
from gfn_composition.tasks.seh_frag import BaseSEHFragTrainer
from gfn_composition.utils.eval_config import (
    frag_config_for_loading,
    qm9_config_for_loading,
)
from gfn_composition.utils.misc import cycle, set_seed
from gfn_composition.utils.postprocess import postprocess_db, write_summary_txt


# Per-objective L/H thresholds for the bin-statistics breakdown in summary.txt.
BIN_THRESHOLDS = {
    "frag": {"seh": 0.5,  "sa": 0.6, "qed": 0.25},
    "qm9":  {"gap": 0.85, "sa": 0.3, "qed": 0.3},
}


def get_args():
    p = argparse.ArgumentParser(description="Evaluate a single-objective base GFN.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir_name", type=str, required=True)
    p.add_argument("--task", type=str, default="frag", choices=["frag", "qm9"])
    p.add_argument("--obj_param", nargs="+", type=str, default=["seh"],
                   choices=["seh", "sa", "qed", "gap"],
                   help="Single objective (list form for API symmetry).")
    p.add_argument("--dist_params", nargs="+", type=float, default=[32.0])
    p.add_argument("--ckpt_path", nargs="+", type=str, required=True,
                   help="Single .pt checkpoint (list form for API symmetry).")
    p.add_argument("--num_from_policy", type=int, default=100)
    p.add_argument("--num_final_gen_steps", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=9)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    if args.task == "qm9":
        config = qm9_config_for_loading(
            dir_name=args.dir_name, obj_param=args.obj_param,
            dist_params=args.dist_params, num_final_gen_steps=args.num_final_gen_steps,
            num_from_policy=args.num_from_policy, seed=args.seed,
            max_nodes=args.max_nodes, device=args.device,
        )
        trainer = BaseQM9Trainer(config, print_config=False)
    else:
        config = frag_config_for_loading(
            dir_name=args.dir_name, obj_param=args.obj_param,
            dist_params=args.dist_params, num_final_gen_steps=args.num_final_gen_steps,
            num_from_policy=args.num_from_policy, seed=args.seed,
            max_nodes=args.max_nodes, device=args.device,
        )
        trainer = BaseSEHFragTrainer(config, print_config=False)

    state_dict = torch.load(args.ckpt_path[0])
    trainer.model.load_state_dict(state_dict["models_state_dict"][0], strict=True)
    trainer.model.to(args.device)

    final_info = {}
    final_dl = trainer.build_final_data_loader()
    for it, batch in zip(range(1, args.num_final_gen_steps + 1), cycle(final_dl)):
        if hasattr(batch, "extra_info"):
            for k, v in batch.extra_info.items():
                final_info.setdefault(k, []).append(v.item() if hasattr(v, "item") else v)
        print(f"  Generating objs {it}/{args.num_final_gen_steps}")
    final_info = {k: float(np.mean(vs)) for k, vs in final_info.items()}
    print("Final generation done — " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))

    # SQLite log uses upstream's fr_0/fr_1 column names; backfill with named columns
    # for every task objective (seh/sa/qed or gap/sa/qed).
    postprocess_db(args.dir_name, args.task, args.dist_params)
    write_summary_txt(args.dir_name, args.task, args.obj_param, args.dist_params,
                      thresholds=BIN_THRESHOLDS[args.task])


if __name__ == "__main__":
    main()
