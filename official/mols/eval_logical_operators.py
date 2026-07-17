"""Compositional mixing with logical-operator-style aggregation.
"""
import argparse
import os

from gfn_composition.algo.mixture_graph_sampling_subtb import make_mixture_sampler_subtb
from gfn_composition.algo.mixture_graph_sampling_tb import make_mixture_sampler_tb
from gfn_composition.algo.mixture_trajectory_balance import MixtureTrajectoryBalance
from gfn_composition.tasks.qm9_moo import MultiObjectiveQM9Trainer
from gfn_composition.tasks.seh_frag_moo import MultiObjectiveSEHFragTrainer
from gfn_composition.utils.eval_config import (
    frag_moo_config_for_loading,
    qm9_moo_config_for_loading,
)
from gfn_composition.utils.eval_loops import build_grid_cond_vectors, run_grid_eval
from gfn_composition.utils.loading import load_ingredient_model_trainers
from gfn_composition.utils.misc import set_seed
from gfn_composition.utils.postprocess import postprocess_db, write_logical_summary_txt


BIN_THRESHOLDS = {
    "frag": {"seh": 0.5, "sa": 0.6, "qed": 0.25},
    "qm9":  {"gap": 0.85, "sa": 0.3, "qed": 0.3},
}


def get_args():
    p = argparse.ArgumentParser(description="Logical-operator mixing over base GFN ingredients.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir_name", type=str, required=True)
    p.add_argument("--task", type=str, default="frag", choices=["frag", "qm9"])
    p.add_argument("--obj_param", nargs="+", type=str, required=True,
                   choices=["seh", "sa", "qed", "gap"],
                   help="Objectives in the same order as --ckpt_path.")
    p.add_argument("--dist_params", nargs="+", type=float, default=[32.0])
    p.add_argument("--ckpt_path", nargs="+", type=str, required=True,
                   help="One base GFN ckpt per objective.")
    p.add_argument("--algo", type=str, default="subtb", choices=["subtb", "tb"],
                   help="TB variant the ingredients were trained with.")
    p.add_argument("--mode", type=str, default="harmonic_mean",
                   choices=["harmonic_mean", "contrast"])
    p.add_argument("--dominant_model_idx", type=int, default=0,
                   help="(contrast only) index of the dominant ingredient (used as p1).")
    p.add_argument("--n_repeat", type=int, default=128,
                   help="Samples for the single uniform-pref bucket.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_nodes", type=int, default=9)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    flow = p.add_mutually_exclusive_group()
    flow.add_argument("--without_u", action="store_true",
                      help="Drop u (the F/Z reaching-prob factor) from the per-ingredient term.")
    flow.add_argument("--use_model_f", action="store_true",
                      help="(default) use F(s) from each ingredient's saved per-graph head.")
    flow.add_argument("--use_db_f", action="store_true",
                      help="Estimate F(s) via the detailed-balance relation log F(s') = log F(s) + log P_F(s'|s) + log n_back.")
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    cfg_fn = qm9_moo_config_for_loading if args.task == "qm9" else frag_moo_config_for_loading
    trainer_cls = MultiObjectiveQM9Trainer if args.task == "qm9" else MultiObjectiveSEHFragTrainer
    config = cfg_fn(
        dir_name=args.dir_name, obj_param=args.obj_param,
        dist_params=args.dist_params, num_final_gen_steps=0,
        num_from_policy=args.batch_size, seed=args.seed,
        max_nodes=args.max_nodes, device=args.device,
    )
    trainer = trainer_cls(config, print_config=False)

    ingredients = load_ingredient_model_trainers(
        ckpt_paths=args.ckpt_path, dist_params=args.dist_params,
        score_names=args.obj_param, task=args.task,
        seed=args.seed, max_nodes=args.max_nodes, device=args.device,
    )

    mixture_algo = MixtureTrajectoryBalance(trainer.env, trainer.ctx, trainer.cfg)
    mixture_algo.task = trainer.task
    sampler_factory = make_mixture_sampler_subtb if args.algo == "subtb" else make_mixture_sampler_tb
    kwargs = dict(mode=args.mode, beta=args.dist_params[0],
                  dominant_model_idx=args.dominant_model_idx,
                  without_u=args.without_u)
    if args.algo == "subtb":
        kwargs["use_db_f"] = args.use_db_f
    mixture_algo.graph_sampler = sampler_factory(mixture_algo, ingredients, **kwargs)
    trainer.algo = mixture_algo
    trainer.model.to(args.device)
    config.num_workers = 0

    # Logical-operator modes are preference-agnostic — single uniform 1/k bucket.
    uniform_pref = [1.0 / len(args.obj_param)] * len(args.obj_param)
    cond_vectors = build_grid_cond_vectors(
        num_objectives=len(args.obj_param),
        pref_mode="fixed",
        fixed_pref=uniform_pref,
        focus_dim=0,
    )
    target_valid = len(cond_vectors) * args.n_repeat
    run_grid_eval(
        trainer, cond_vectors,
        n_repeat=args.n_repeat, batch_size=args.batch_size,
        output_dir=os.path.join(args.dir_name, "final"),
        mixture_mode=True,
        target_valid=target_valid,
    )
    # The sampler doesn't write per-objective scores; backfill all of them.
    postprocess_db(args.dir_name, args.task, args.dist_params)
    write_logical_summary_txt(args.dir_name, args.task, args.obj_param, args.dist_params,
                              thresholds=BIN_THRESHOLDS[args.task])


if __name__ == "__main__":
    main()
