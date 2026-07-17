"""Single-objective fragment-based GFlowNet on SEH binding (or QED / SA).
"""
import argparse
from typing import List, Tuple

import torch
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties
from gflownet.algo.config import TBVariant
from gflownet.config import Config, init_empty
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer, SEHTask

from gfn_composition.tasks._aux import AUX_TASKS


class BaseSEHTask(SEHTask):
    """Single-objective SEH-frag task that supports `seh`, `qed`, or `sa` as the reward.

    For `seh`, the reward comes from the pre-trained Bengio-2021 binding-energy proxy
    inherited from upstream. For `qed`/`sa`, the molecule is scored directly via `AUX_TASKS`.
    """

    def __init__(self, cfg, wrap_model=None):
        super().__init__(cfg, wrap_model)
        # `cfg.task.seh_moo.objectives` is a list of length 1 in single-objective mode
        # (we reuse the MOO config field so the same upstream classes work).
        self.objectives = cfg.task.seh_moo.objectives

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        obj = self.objectives[0]
        if obj == "seh":
            preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        else:
            preds = AUX_TASKS[obj](mols, is_valid).reshape((-1, 1))

        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class BaseSEHFragTrainer(SEHFragTrainer):
    """Trainer for `BaseSEHTask`. Inherits all setup from upstream `SEHFragTrainer`,
    only swapping the task class so the non-`seh` objective routing in `compute_obj_properties`
    is picked up."""

    def setup_task(self):
        self.task = BaseSEHTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)


def get_args():
    parser = argparse.ArgumentParser(description="Train SEH fragment GFlowNet (single objective).")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--objectives", nargs="+", type=str, default=["seh"],
                        choices=["seh", "qed", "sa"])
    parser.add_argument("--dist_params", nargs="+", type=float, default=[32.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tb_variant", type=str, default="subtb", choices=["tb", "subtb", "db"])
    parser.add_argument("--num_training_steps", type=int, default=15_000)
    return parser.parse_args()


def main():
    args = get_args()
    assert len(args.objectives) == 1, (
        f"train_gfn.py seh_frag trains a single-objective GFN; got --objectives {args.objectives}"
    )

    config = init_empty(Config())
    config.log_dir = args.log_dir
    config.task.seh_moo.objectives = args.objectives
    config.cond.temperature.dist_params = args.dist_params
    config.cond.temperature.sample_dist = "constant"
    config.algo.tb.do_parameterize_p_b = False
    config.algo.max_nodes = 9
    config.seed = args.seed

    config.device = "cuda"
    config.num_workers = 8
    config.print_every = 500
    config.validate_every = 5000
    config.num_final_gen_steps = 10
    config.num_training_steps = args.num_training_steps
    config.overwrite_existing_exp = False
    config.algo.num_from_policy = 64
    config.algo.num_from_dataset = 0

    config.model.num_emb = 128
    config.model.num_layers = 6

    config.opt.learning_rate = 5e-4
    config.algo.sampling_tau = 0.99
    config.algo.train_random_action_prob = 0.05
    config.algo.tb.Z_learning_rate = 5e-4
    config.algo.illegal_action_logreward = -75
    config.algo.tb.variant = {"tb": TBVariant.TB, "subtb": TBVariant.SubTB1, "db": TBVariant.DB}[args.tb_variant]

    trial = BaseSEHFragTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
