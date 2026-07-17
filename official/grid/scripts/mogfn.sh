#!/bin/bash
# Train MO-GFN with replay buffer + evaluate

python train_mogfn.py \
    --device cuda:6 \
    --custom_dist mix2 --n_reward_fns 2 \
    --replay_buffer_size 10000 \
    --save_dir ./pretrained_models

python eval_mogfn.py \
    --model_saved_path pretrained_models/mogfn.pt \
    --custom_dist mix2 --n_reward_fns 2 \
    --result_saved_path ./pretrained_models/eval_result.png \
    --device cuda:0
