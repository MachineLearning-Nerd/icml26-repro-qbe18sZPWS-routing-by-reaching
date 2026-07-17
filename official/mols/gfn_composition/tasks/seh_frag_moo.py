"""Multi-objective fragment-based GFlowNet wrapper for SEH binding + auxiliary objectives.
"""
import pathlib
from typing import List, Tuple

import torch
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader

from gflownet import ObjectProperties
from gflownet.data.data_source import DataSource
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag_moo import SEHMOOFragTrainer, SEHMOOTask

from gfn_composition.tasks._aux import AUX_TASKS
from gfn_composition.utils.sqlite_log import NamedColumnSQLiteLogHook


class MultiObjectiveSEHTask(SEHMOOTask):
    def __init__(self, cfg, wrap_model=None):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"seh", "qed", "sa"}
        assert len(self.objectives) == len(set(self.objectives))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)
        if not any(is_valid):
            return ObjectProperties(torch.zeros((0, len(self.objectives)))), is_valid_t

        flat_r: List[Tensor] = []
        for obj in self.objectives:
            if obj == "seh":
                flat_r.append(super().compute_reward_from_graph(graphs))
            else:
                flat_r.append(AUX_TASKS[obj](mols, is_valid))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(mols)
        return ObjectProperties(flat_rewards), is_valid_t


class MultiObjectiveSEHFragTrainer(SEHMOOFragTrainer):
    def setup_task(self):
        self.cfg.cond.moo.num_objectives = len(self.cfg.task.seh_moo.objectives)
        self.task = MultiObjectiveSEHTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

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
