"""TrajectoryBalance subclass that forwards the **full** cond_info dict to the
graph sampler when the sampler is a `MixtureGraphSampler` / `MixtureGraphSamplerTB`.

Mixture samplers need access to `cond_info["preferences"]` (raw scalarization weights)
which would otherwise be lost: upstream's `create_training_data_from_own_samples` only
takes the encoded tensor (`cond_info["encoding"]`).

Pair this with `MixtureDataSource` (which preserves the dict and forwards it as
`total_cond_info`).
"""
from typing import Optional

from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalance, TrajectoryBalanceModel
from gflownet.utils.misc import get_worker_device


class MixtureTrajectoryBalance(TrajectoryBalance):
    def create_training_data_from_own_samples(
        self,
        model: TrajectoryBalanceModel,
        n: int,
        cond_info: Optional[Tensor] = None,
        random_action_prob: Optional[float] = 0.0,
        total_cond_info: Optional[dict] = None,
    ):
        """Same as upstream, but passes `total_cond_info` (the full cond_info dict)
        through to `graph_sampler.sample_from_model`. The mixture samplers read
        `total_cond_info["preferences"]` to weight ingredients."""
        dev = get_worker_device()
        cond_info = cond_info.to(dev) if cond_info is not None else None
        data = self.graph_sampler.sample_from_model(
            model, n, cond_info, random_action_prob, total_cond_info=total_cond_info,
        )
        if cond_info is not None:
            logZ_pred = model.logZ(cond_info)
            for i in range(n):
                data[i]["logZ"] = logZ_pred[i].item()
        return data
