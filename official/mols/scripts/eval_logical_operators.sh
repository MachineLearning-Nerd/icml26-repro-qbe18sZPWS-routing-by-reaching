#!/usr/bin/env bash
# Mixing eval — logical operators (HM + contrast). Representative commands across
# mode (HM / contrast) × MIX-size (MIX2 / MIX3) × F-variant (model_f / db_f).
# 5000 mols/run. Run from `mols/`.

# ============== Harmonic mean (HM) ==============

# HM MIX2 (model_f)
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa     --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt                                                              --dir_name ./eval_results/mixing/frag/subtb/model_f/harmonic_MIX2_seh_sa_seed604     --n_repeat 5000 --batch_size 125 --algo subtb --mode harmonic_mean --use_model_f --device cuda --seed 604

# HM MIX2 (db_f)
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa     --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt                                                              --dir_name ./eval_results/mixing/frag/subtb/db_f/harmonic_MIX2_seh_sa_seed604        --n_repeat 5000 --batch_size 125 --algo subtb --mode harmonic_mean --use_db_f    --device cuda --seed 604

# HM MIX3 (model_f)
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt          --dir_name ./eval_results/mixing/frag/subtb/model_f/harmonic_MIX3_seh_sa_qed_seed604 --n_repeat 5000 --batch_size 125 --algo subtb --mode harmonic_mean --use_model_f --device cuda --seed 604

# HM MIX3 (db_f)
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt          --dir_name ./eval_results/mixing/frag/subtb/db_f/harmonic_MIX3_seh_sa_qed_seed604    --n_repeat 5000 --batch_size 125 --algo subtb --mode harmonic_mean --use_db_f    --device cuda --seed 604

# ============== Contrast (CT) — --dominant_model_idx picks the dominant ingredient ==============

# CT MIX2 (model_f) — seh over sa
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa     --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt                                                              --dir_name ./eval_results/mixing/frag/subtb/model_f/contrast_MIX2_seh_over_sa_seed604 --n_repeat 5000 --batch_size 125 --algo subtb --mode contrast --dominant_model_idx 0 --use_model_f --device cuda --seed 604

# CT MIX2 (db_f) — seh over sa
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa     --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt                                                              --dir_name ./eval_results/mixing/frag/subtb/db_f/contrast_MIX2_seh_over_sa_seed604    --n_repeat 5000 --batch_size 125 --algo subtb --mode contrast --dominant_model_idx 0 --use_db_f    --device cuda --seed 604

# CT MIX3 (model_f) — seh dominant
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt          --dir_name ./eval_results/mixing/frag/subtb/model_f/contrast_MIX3_seh_dom_seed604     --n_repeat 5000 --batch_size 125 --algo subtb --mode contrast --dominant_model_idx 0 --use_model_f --device cuda --seed 604

# CT MIX3 (db_f) — seh dominant
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt          --dir_name ./eval_results/mixing/frag/subtb/db_f/contrast_MIX3_seh_dom_seed604        --n_repeat 5000 --batch_size 125 --algo subtb --mode contrast --dominant_model_idx 0 --use_db_f    --device cuda --seed 604

# CT MIX2 — sa over seh (dominant_model_idx=1)
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa     --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt                                                              --dir_name ./eval_results/mixing/frag/subtb/model_f/contrast_MIX2_sa_over_seh_seed604  --n_repeat 5000 --batch_size 125 --algo subtb --mode contrast --dominant_model_idx 1 --use_model_f --device cuda --seed 604

# CT MIX3 — qed dominant (dominant_model_idx=2)
CUDA_VISIBLE_DEVICES=0 python eval_logical_operators.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt          --dir_name ./eval_results/mixing/frag/subtb/model_f/contrast_MIX3_qed_dom_seed604      --n_repeat 5000 --batch_size 125 --algo subtb --mode contrast --dominant_model_idx 2 --use_model_f --device cuda --seed 604
