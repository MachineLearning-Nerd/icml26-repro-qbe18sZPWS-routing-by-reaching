#!/usr/bin/env bash
# Train 6 single-objective base GFNs (frag × {seh, sa, qed} + qm9 × {gap, sa, qed})
# at seed 604 with SubTB. These are the ingredient models that the mixing-eval

CUDA_VISIBLE_DEVICES=0 python train_gfn.py seh_frag --log_dir ./ckpts/base-gfn/frag/subtb/seh_seed604 --objectives seh --tb_variant subtb --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=1 python train_gfn.py seh_frag --log_dir ./ckpts/base-gfn/frag/subtb/sa_seed604  --objectives sa  --tb_variant subtb --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=2 python train_gfn.py seh_frag --log_dir ./ckpts/base-gfn/frag/subtb/qed_seed604 --objectives qed --tb_variant subtb --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=3 python train_gfn.py qm9      --log_dir ./ckpts/base-gfn/qm9/subtb/gap_seed604  --objectives gap --tb_variant subtb --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=4 python train_gfn.py qm9      --log_dir ./ckpts/base-gfn/qm9/subtb/sa_seed604   --objectives sa  --tb_variant subtb --seed 604 &
sleep 3
CUDA_VISIBLE_DEVICES=5 python train_gfn.py qm9      --log_dir ./ckpts/base-gfn/qm9/subtb/qed_seed604  --objectives qed --tb_variant subtb --seed 604 &
sleep 3
wait

echo "Done: train/base_gfn_seed604.sh (6 jobs)"
