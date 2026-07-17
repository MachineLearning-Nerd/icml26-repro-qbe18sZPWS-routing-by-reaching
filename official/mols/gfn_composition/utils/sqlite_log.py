"""SQLiteLogHook variant that uses meaningful column names for objective properties.

Upstream's SQLiteLogHook labels per-objective columns as `fr_0, fr_1, ...`. This subclass
takes an explicit `obj_names` list (e.g. ["gap", "sa", "qed"]) and uses those as column
names instead, which makes downstream analysis much easier.
"""
import os
from typing import List, Optional

import torch

from gflownet.utils.sqlite_log import SQLiteLog, SQLiteLogHook


class NamedColumnSQLiteLogHook(SQLiteLogHook):
    """SQLiteLogHook that writes obj_props columns under `obj_names` instead of `fr_*`.

    If `obj_names` is None or its length doesn't match the number of objectives, falls back
    to the upstream `fr_0, fr_1, ...` labels.
    """

    def __init__(self, log_dir, ctx, obj_names: Optional[List[str]] = None) -> None:
        super().__init__(log_dir, ctx)
        self.obj_names = obj_names

    def __call__(self, trajs, rewards, obj_props, cond_info):
        if self.log is None:
            worker_info = torch.utils.data.get_worker_info()
            self._wid = worker_info.id if worker_info is not None else 0
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_objs_{self._wid}.db"
            self.log = SQLiteLog()
            self.log.connect(self.log_path)

        if hasattr(self.ctx, "object_to_log_repr"):
            objs = [self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else "" for t in trajs]
        else:
            objs = [""] * len(trajs)

        obj_props = obj_props.reshape((len(obj_props), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(objs), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(objs), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [objs[i], rewards[i]]
            + obj_props[i]
            + preferences[i]
            + focus_dir[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]

        if self.data_labels is None:
            n_obj = len(obj_props[0]) if obj_props else 0
            if self.obj_names and len(self.obj_names) == n_obj:
                obj_cols = list(self.obj_names)
            else:
                obj_cols = [f"fr_{i}" for i in range(n_obj)]
            self.data_labels = (
                ["smi", "r"]
                + obj_cols
                + [f"pref_{i}" for i in range(len(preferences[0]))]
                + [f"focus_{i}" for i in range(len(focus_dir[0]))]
                + [f"ci_{k}" for k in logged_keys]
            )

        self.log.insert_many(data, self.data_labels)
        # Commit so that other connections (e.g. _count_valid_in_db during retry) see the rows.
        self.log.db.commit()
        return {}
