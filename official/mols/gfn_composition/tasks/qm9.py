"""Single-objective atom-by-atom GFlowNet on QM9 (gap, qed, or sa).
"""
import argparse
from typing import List, Tuple

import torch
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

import gflownet.models.mxmnet as mxmnet
from gflownet import ObjectProperties
from gflownet.algo.config import TBVariant
from gflownet.config import Config, init_empty
from gflownet.tasks.qm9 import QM9GapTask, QM9GapTrainer

from gfn_composition.data.qm9_dataset import QM9TransformedDataset
from gfn_composition.tasks._aux import AUX_TASKS


class BaseQM9Task(QM9GapTask):
    """Single-objective QM9 task that supports `gap`, `qed`, or `sa` as the reward.

    For `gap`, the reward goes through the MXMNet proxy + percentile transform inherited
    from upstream. For `qed`/`sa`, the molecule is scored directly via `AUX_TASKS`.
    """

    def __init__(self, dataset, cfg, wrap_model=None):
        super().__init__(dataset, cfg, wrap_model)
        # `cfg.task.qm9_moo.objectives` is a list of length 1 in single-objective mode
        # (we reuse the MOO config field so the same upstream classes work). See `main()`.
        self.objectives = cfg.task.qm9_moo.objectives

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        obj = self.objectives[0]
        if obj == "gap":
            preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        else:
            preds = AUX_TASKS[obj](mols, is_valid).reshape((-1, 1))

        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class BaseQM9Trainer(QM9GapTrainer):
    def setup_data(self):
        targets = list(self.cfg.task.qm9_moo.objectives)
        self.training_data = QM9TransformedDataset(self.cfg.task.qm9.h5_path, train=True, targets=targets)
        self.test_data = QM9TransformedDataset(self.cfg.task.qm9.h5_path, train=False, targets=targets)
        self.to_terminate.append(self.training_data.terminate)
        self.to_terminate.append(self.test_data.terminate)

    def setup_task(self):
        self.task = BaseQM9Task(
            dataset=self.training_data,
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )


def get_args():
    parser = argparse.ArgumentParser(description="Train single-objective GFlowNet on QM9.")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--objectives", nargs="+", type=str, default=["qed"],
                        choices=["gap", "qed", "sa"])
    parser.add_argument("--dist_params", nargs="+", type=float, default=[32.0])
    parser.add_argument("--h5_path", type=str, default="./data/qm9.h5")
    parser.add_argument("--mxmnet_path", type=str, default="./data/mxmnet_gap_model.pt",
                        help="Path to mxmnet_gap_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tb_variant", type=str, default="subtb", choices=["tb", "subtb", "db"])
    parser.add_argument("--num_training_steps", type=int, default=15_000)
    parser.add_argument("--num_from_policy", type=int, default=64)
    parser.add_argument("--num_from_dataset", type=int, default=64)
    return parser.parse_args()


def main():
    args = get_args()
    assert len(args.objectives) == 1, (
        f"train_gfn.py qm9 trains a single-objective GFN; got --objectives {args.objectives}"
    )

    config = init_empty(Config())
    config.log_dir = args.log_dir
    config.task.qm9_moo.objectives = args.objectives
    config.task.qm9.h5_path = args.h5_path
    config.task.qm9.model_path = args.mxmnet_path
    config.cond.temperature.dist_params = args.dist_params
    config.cond.temperature.sample_dist = "constant"
    config.algo.tb.do_parameterize_p_b = False
    config.seed = args.seed

    config.device = "cuda"
    config.num_workers = 8
    config.print_every = 300
    config.validate_every = 5000
    config.num_validation_gen_steps = 15
    config.num_final_gen_steps = 0
    config.num_training_steps = args.num_training_steps
    config.overwrite_existing_exp = False

    config.model.num_emb = 64
    config.model.num_layers = 4
    config.algo.num_from_policy = args.num_from_policy
    config.algo.num_from_dataset = args.num_from_dataset

    config.cond.focus_region.focus_type = None
    config.cond.valid_sample_cond_info = False

    config.opt.learning_rate = 5e-4
    config.algo.tb.Z_learning_rate = 5e-4
    config.algo.train_random_action_prob = 0.05
    config.algo.sampling_tau = 0.99
    config.algo.illegal_action_logreward = -75
    config.algo.tb.variant = {"tb": TBVariant.TB, "subtb": TBVariant.SubTB1, "db": TBVariant.DB}[args.tb_variant]

    trial = BaseQM9Trainer(config)
    trial.run()


if __name__ == "__main__":
    main()
