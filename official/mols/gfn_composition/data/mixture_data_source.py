"""DataSource subclass that passes the full cond_info dict to the algo.

Used together with `MixtureTrajectoryBalance` so that mixture samplers can read
`cond_info["preferences"]`, which the upstream `DataSource` discards (passes only
`cond_info["encoding"]`).

We override the three iterators that call `create_training_data_from_own_samples`:
  - `do_sample_model`            — used by online training / inference
  - `do_sample_model_n_times`    — used by `build_final_data_loader`
  - `do_conditionals_dataset_in_order` — used by grid eval
"""
import torch

from gflownet.data.data_source import DataSource


class MixtureDataSource(DataSource):
    def do_sample_model(self, model, num_from_policy, num_new_replay_samples=None):
        """Online sampling iterator. Identical to upstream except it forwards the full
        `cond_info` dict (not just `cond_info["encoding"]`) to `create_training_data_from_own_samples`."""
        if num_new_replay_samples is not None:
            assert self.replay_buffer is not None, "num_new_replay_samples specified without a replay buffer"
        if num_new_replay_samples is None:
            assert self.replay_buffer is None, "num_new_replay_samples not specified with a replay buffer"
        num_new_replay_samples = num_new_replay_samples or 0
        num_samples = max(num_from_policy, num_new_replay_samples)

        def iterator():
            while self.active:
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(num_samples, t)
                trajs = self.algo.create_training_data_from_own_samples(
                    model, num_samples, cond_info["encoding"], p, cond_info,
                )
                self.set_traj_cond_info(trajs, cond_info)
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs[:num_new_replay_samples])
                batch_info = self.call_sampling_hooks(trajs)
                yield (trajs[:num_from_policy], batch_info)

        self.iterators.append(iterator)
        return self

    def do_sample_model_n_times(self, model, num_samples_per_batch, num_total):
        """Bounded sampling iterator (stop after `num_total`). Same upstream behavior with the
        full cond_info dict forwarded to `create_training_data_from_own_samples`."""
        total = torch.zeros(1, dtype=torch.int64)
        total.share_memory_()
        total_lock = torch.multiprocessing.Lock()
        total_barrier = torch.multiprocessing.Barrier(max(1, self.cfg.num_workers))

        def iterator():
            while self.active:
                with total_lock:
                    n_so_far = total.item()
                    n_this_time = min(num_total - n_so_far, num_samples_per_batch)
                    total[:] += n_this_time
                    if n_this_time == 0:
                        break
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.sample_conditional_information(n_this_time, t)
                trajs = self.algo.create_training_data_from_own_samples(
                    model, n_this_time, cond_info["encoding"], p, cond_info,
                )
                self.set_traj_cond_info(trajs, cond_info)
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                batch_info = self.call_sampling_hooks(trajs)
                yield (trajs, batch_info)
            total_barrier.wait()
            total[:] = 0

        self.iterators.append(iterator)
        return self

    def do_conditionals_dataset_in_order(self, data, num_samples, model):
        """Grid-eval iterator: sweep `data` (a `RepeatedCondInfoDataset` of preference vectors)
        and sample one batch per chunk, forwarding the full cond_info dict so the mixture
        sampler can read `preferences`. This is the path used by `run_grid_eval`."""
        def iterator():
            for idcs in self.iterate_indices(len(data), num_samples):
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                cond_info = self.task.encode_conditional_information(
                    torch.stack([data[i] for i in idcs])
                )
                trajs = self.algo.create_training_data_from_own_samples(
                    model, len(idcs), cond_info["encoding"], p, cond_info,
                )
                self.set_traj_cond_info(trajs, cond_info)
                self.compute_properties(trajs, mark_as_online=True)
                self.compute_log_rewards(trajs)
                self.send_to_replay(trajs)
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
                batch_info = self.call_sampling_hooks(trajs)
                yield (trajs, batch_info)

        self.iterators.append(iterator)
        return self
