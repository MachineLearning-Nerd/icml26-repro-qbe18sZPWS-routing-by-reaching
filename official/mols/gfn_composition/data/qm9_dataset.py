import torch

from gflownet.data.qm9 import QM9Dataset


class QM9TransformedDataset(QM9Dataset):
    def setup(self, task, ctx):
        super().setup(task, ctx)
        self.task = task

    def _transform_targets(self, raw_targets):
        transformed = torch.zeros_like(raw_targets)
        for i, t in enumerate(self.targets):
            if t == "gap":
                transformed[i] = (1 - (raw_targets[i] - self.task._percentile_95) / self.task._width).clip(1e-4, 2)
            elif t == "sa":
                transformed[i] = (10 - raw_targets[i]) / 9
            else:
                transformed[i] = raw_targets[i]
        return transformed

    def __getitem__(self, idx):
        graph, raw_targets = super().__getitem__(idx)
        if getattr(self, "task", None) is not None:
            raw_targets = self._transform_targets(raw_targets)
        return graph, raw_targets
