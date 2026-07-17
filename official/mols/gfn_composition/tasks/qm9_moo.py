"""Multi-objective atom-by-atom GFlowNet wrapper for QM9 (gap, qed, sa).
"""
import pathlib
from typing import List, Tuple

import torch
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader

import gflownet.models.mxmnet as mxmnet
from gflownet import ObjectProperties
from gflownet.data.data_source import DataSource
from gflownet.tasks.qm9_moo import QM9GapMOOTask, QM9MOOTrainer

from gfn_composition.data.qm9_dataset import QM9TransformedDataset
from gfn_composition.tasks._aux import AUX_TASKS
from gfn_composition.utils.sqlite_log import NamedColumnSQLiteLogHook


class MultiObjectiveQM9Task(QM9GapMOOTask):
    def __init__(self, dataset, cfg, wrap_model=None):
        super().__init__(dataset, cfg, wrap_model)
        assert set(self.objectives) <= {"gap", "qed", "sa"}
        assert len(self.objectives) == len(set(self.objectives))

    def reward_transform(self, y):
        # Upstream `QM9GapMOOTask.reward_transform` is identity (returns raw Hartree gap), which
        # makes "smaller raw value → smaller reward → minimize gap". Single-objective `QM9GapTask`
        # uses `1 - (y - p95)/width` (i.e. "smaller raw value → larger reward → maximize gap").
        # We restore the base GFN's transform so the multi-objective trainer (used to set up
        # mixing eval) learns gap in the same direction as the base GFN, and so postprocessed
        # `gap` columns match `fr_0`.
        from gflownet.tasks.qm9 import QM9GapTask
        return QM9GapTask.reward_transform(self, y)

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        if not any(is_valid):
            return ObjectProperties(torch.zeros((0, len(self.objectives)))), is_valid_t

        flat_r: List[Tensor] = []
        for obj in self.objectives:
            if obj == "gap":
                flat_r.append(super().compute_reward_from_graph(graphs))
            else:
                flat_r.append(AUX_TASKS[obj](mols, is_valid))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == is_valid_t.sum()
        return ObjectProperties(flat_rewards), is_valid_t


class MultiObjectiveQM9Trainer(QM9MOOTrainer):
    def setup_task(self):
        self.task = MultiObjectiveQM9Task(
            dataset=self.training_data,
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_data(self):
        self.training_data = QM9TransformedDataset(
            self.cfg.task.qm9.h5_path, train=True, targets=self.cfg.task.qm9_moo.objectives
        )
        self.test_data = QM9TransformedDataset(
            self.cfg.task.qm9.h5_path, train=False, targets=self.cfg.task.qm9_moo.objectives
        )
        self.to_terminate.append(self.training_data.terminate)
        self.to_terminate.append(self.test_data.terminate)

    def build_validation_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        src.do_conditionals_dataset_in_order(self.test_data, self.cfg.algo.valid_num_from_dataset, model)
        if self.cfg.log_dir:
            src.add_sampling_hook(
                NamedColumnSQLiteLogHook(
                    str(pathlib.Path(self.cfg.log_dir) / "valid"),
                    self.ctx,
                    obj_names=self.task.objectives,
                )
            )
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)
