"""Config builders for **eval / inference time only**.

Build a minimal config sufficient to:
  - Instantiate a Trainer (which sets up env, ctx, model, algo, task).
  - Load a saved checkpoint into `trainer.model`.
  - Run sampling (`do_sample_model_n_times`) and post-processing.

Training-only fields (learning_rate, sampling_tau, train_random_action_prob,
Z_learning_rate, num_training_steps, etc.) are intentionally omitted — they do
not affect model architecture or the sampling pass. The only fields kept are
those that:
  - Affect `model.state_dict()` shape (model arch, do_parameterize_p_b, num_cond_dim).
  - Affect the sampling pass (max_nodes, num_from_policy, num_final_gen_steps, dist_params).
  - Identify input data (h5 path, mxmnet weights).
"""
from gflownet.config import Config, init_empty


def _config_for_loading(dir_name, dist_params, num_from_policy, num_final_gen_steps,
                        device, max_nodes, seed):
    """Minimum config to instantiate a trainer + load a ckpt + sample."""
    config = init_empty(Config())
    config.log_dir = dir_name
    config.seed = seed
    config.device = device
    config.num_workers = 0

    config.algo.max_nodes = max_nodes
    config.algo.num_from_policy = num_from_policy
    config.algo.tb.do_parameterize_p_b = False  # arch-affecting; matches our training default

    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = dist_params

    config.num_final_gen_steps = num_final_gen_steps
    return config


def _apply_qm9(config, obj_param):
    config.task.qm9_moo.objectives = obj_param
    config.model.num_emb = 64
    config.model.num_layers = 4
    # Scripts assume cwd = mols/ (see scripts/{base-gfn,mixing}/*.sh). These relative paths
    # resolve to mols/data/* under that assumption.
    config.task.qm9.h5_path = "./data/qm9.h5"
    config.task.qm9.model_path = "./data/mxmnet_gap_model.pt"


def _apply_frag(config, obj_param):
    config.task.seh_moo.objectives = obj_param
    config.model.num_emb = 128
    config.model.num_layers = 6


def _apply_moo(config):
    # focus_type=None is REQUIRED: any other value (incl. upstream default "centered")
    # adds focus_cond encoding dims to `task.num_cond_dim`, which changes the first-layer
    # shape of the model and would break ingredient-ckpt loading.
    config.cond.focus_region.focus_type = None


def qm9_config_for_loading(dir_name="temp", obj_param=("gap",), dist_params=(32.0,),
                           num_from_policy=10, num_final_gen_steps=128,
                           device="cpu", max_nodes=9, seed=0):
    config = _config_for_loading(dir_name, list(dist_params), num_from_policy,
                                 num_final_gen_steps, device, max_nodes, seed)
    _apply_qm9(config, list(obj_param))
    return config


def qm9_moo_config_for_loading(dir_name="temp", obj_param=("gap", "qed", "sa"),
                                dist_params=(32.0,), num_from_policy=10, num_final_gen_steps=128,
                                device="cpu", max_nodes=9, seed=0):
    config = _config_for_loading(dir_name, list(dist_params), num_from_policy,
                                 num_final_gen_steps, device, max_nodes, seed)
    _apply_qm9(config, list(obj_param))
    _apply_moo(config)
    return config


def frag_config_for_loading(dir_name="tmp", obj_param=("seh",), dist_params=(32.0,),
                            num_from_policy=10, num_final_gen_steps=128,
                            device="cpu", max_nodes=9, seed=0):
    config = _config_for_loading(dir_name, list(dist_params), num_from_policy,
                                 num_final_gen_steps, device, max_nodes, seed)
    _apply_frag(config, list(obj_param))
    return config


def frag_moo_config_for_loading(dir_name="tmp", obj_param=("seh",),
                                 dist_params=(32.0,), num_from_policy=10, num_final_gen_steps=128,
                                 device="cpu", max_nodes=9, seed=0):
    config = _config_for_loading(dir_name, list(dist_params), num_from_policy,
                                 num_final_gen_steps, device, max_nodes, seed)
    _apply_frag(config, list(obj_param))
    _apply_moo(config)
    return config
