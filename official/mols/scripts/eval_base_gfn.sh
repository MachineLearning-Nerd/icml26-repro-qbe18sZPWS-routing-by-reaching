#!/usr/bin/env bash
# Generate molecules from each base GFN ingredient (seed 604) and write per-objective

CUDA_VISIBLE_DEVICES=0 python eval_gfn.py --task frag --obj_param seh --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/seh_seed604/model_state.pt --dir_name ./eval_results/base-gfn/frag/subtb/seh_seed604 --num_from_policy 100 --num_final_gen_steps 50 --device cuda --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=1 python eval_gfn.py --task frag --obj_param sa  --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/sa_seed604/model_state.pt  --dir_name ./eval_results/base-gfn/frag/subtb/sa_seed604  --num_from_policy 100 --num_final_gen_steps 50 --device cuda --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=2 python eval_gfn.py --task frag --obj_param qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/frag/subtb/qed_seed604/model_state.pt --dir_name ./eval_results/base-gfn/frag/subtb/qed_seed604 --num_from_policy 100 --num_final_gen_steps 50 --device cuda --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=3 python eval_gfn.py --task qm9  --obj_param gap --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/gap_seed604/model_state.pt  --dir_name ./eval_results/base-gfn/qm9/subtb/gap_seed604  --num_from_policy 100 --num_final_gen_steps 50 --device cuda --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=4 python eval_gfn.py --task qm9  --obj_param sa  --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/sa_seed604/model_state.pt   --dir_name ./eval_results/base-gfn/qm9/subtb/sa_seed604   --num_from_policy 100 --num_final_gen_steps 50 --device cuda --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=5 python eval_gfn.py --task qm9  --obj_param qed --dist_params 32.0 --ckpt_path ./ckpts/base-gfn/qm9/subtb/qed_seed604/model_state.pt  --dir_name ./eval_results/base-gfn/qm9/subtb/qed_seed604  --num_from_policy 100 --num_final_gen_steps 50 --device cuda --seed 604 &
sleep 3
wait

echo "Done: eval/base_gfn_seed604.sh (6 jobs)"
