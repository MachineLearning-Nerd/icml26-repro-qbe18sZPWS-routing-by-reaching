#!/bin/bash
# Train HN-GFN with replay buffer

python train_hngfn.py \
    --device cuda:0 \
    --custom_dist mix2 --n_reward_fns 2 \
    --replay_buffer_size 10000 \
    --save_dir ./pretrained_models/hngfns

python eval_hngfn.py \
    --model_saved_path pretrained_models/hngfns/hngfn.pt \
    --custom_dist mix2 --n_reward_fns 2 \
    --result_saved_path ./pretrained_models/hngfns/eval_reasult.png \
    --device cuda:0
