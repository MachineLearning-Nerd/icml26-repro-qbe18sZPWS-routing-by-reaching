#!/usr/bin/env bash
# Mixing eval — scalarization. Representative commands across
# task (frag / qm9) × MIX-size (MIX2 / MIX3) × F-variant (model_f / db_f / without_u).
# MIX2 uses n_pref=10 × n_repeat=128 = 1280 mols; MIX3 uses n_pref=128 × n_repeat=128 = 16384 mols.
# Run from `mols/`.

# ============== frag MIX2 ==============

# model_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task frag --obj_param seh sa --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt --dir_name ./eval_results/mixing/frag/subtb/model_f/scalarization_MIX2_seh_sa_seed604     --pref_mode simplex --n_pref 10 --n_repeat 128 --batch_size 128 --algo subtb --use_model_f --device cuda --seed 604

# db_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task frag --obj_param seh sa --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt --dir_name ./eval_results/mixing/frag/subtb/db_f/scalarization_MIX2_seh_sa_seed604        --pref_mode simplex --n_pref 10 --n_repeat 128 --batch_size 128 --algo subtb --use_db_f    --device cuda --seed 604

# without_u
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task frag --obj_param seh sa --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt --dir_name ./eval_results/mixing/frag/subtb/without_u/scalarization_MIX2_seh_sa_seed604   --pref_mode simplex --n_pref 10 --n_repeat 128 --batch_size 128 --algo subtb --without_u   --device cuda --seed 604

# ============== frag MIX3 ==============

# model_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/mixing/frag/subtb/model_f/scalarization_MIX3_seh_sa_qed_seed604     --pref_mode simplex --n_pref 128 --n_repeat 128 --batch_size 128 --algo subtb --use_model_f --device cuda --seed 604

# db_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/mixing/frag/subtb/db_f/scalarization_MIX3_seh_sa_qed_seed604        --pref_mode simplex --n_pref 128 --n_repeat 128 --batch_size 128 --algo subtb --use_db_f    --device cuda --seed 604

# without_u
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task frag --obj_param seh sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/mixing/frag/subtb/without_u/scalarization_MIX3_seh_sa_qed_seed604   --pref_mode simplex --n_pref 128 --n_repeat 128 --batch_size 128 --algo subtb --without_u   --device cuda --seed 604

# ============== qm9 MIX2 ==============

# model_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task qm9 --obj_param gap sa --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt --dir_name ./eval_results/mixing/qm9/subtb/model_f/scalarization_MIX2_gap_sa_seed604     --pref_mode simplex --n_pref 10 --n_repeat 128 --batch_size 128 --algo subtb --use_model_f --device cuda --seed 604

# db_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task qm9 --obj_param gap sa --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt --dir_name ./eval_results/mixing/qm9/subtb/db_f/scalarization_MIX2_gap_sa_seed604        --pref_mode simplex --n_pref 10 --n_repeat 128 --batch_size 128 --algo subtb --use_db_f    --device cuda --seed 604

# without_u
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task qm9 --obj_param gap sa --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt --dir_name ./eval_results/mixing/qm9/subtb/without_u/scalarization_MIX2_gap_sa_seed604   --pref_mode simplex --n_pref 10 --n_repeat 128 --batch_size 128 --algo subtb --without_u   --device cuda --seed 604

# ============== qm9 MIX3 ==============

# model_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task qm9 --obj_param gap sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/mixing/qm9/subtb/model_f/scalarization_MIX3_gap_sa_qed_seed604     --pref_mode simplex --n_pref 128 --n_repeat 128 --batch_size 128 --algo subtb --use_model_f --device cuda --seed 604

# db_f
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task qm9 --obj_param gap sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/mixing/qm9/subtb/db_f/scalarization_MIX3_gap_sa_qed_seed604        --pref_mode simplex --n_pref 128 --n_repeat 128 --batch_size 128 --algo subtb --use_db_f    --device cuda --seed 604

# without_u
CUDA_VISIBLE_DEVICES=0 python eval_scalarization.py --task qm9 --obj_param gap sa qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt ./ckpts/base-gfn/qm9/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/mixing/qm9/subtb/without_u/scalarization_MIX3_gap_sa_qed_seed604   --pref_mode simplex --n_pref 128 --n_repeat 128 --batch_size 128 --algo subtb --without_u   --device cuda --seed 604
