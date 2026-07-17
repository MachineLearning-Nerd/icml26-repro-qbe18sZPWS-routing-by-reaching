#!/bin/bash
# Train base GFlowNet examples

# Standard
python train_gfn.py \
    --device cuda:0 \
    --custom_dist shubert \
    --save_dir ./test

# With beta (reward sharpening)
python train_gfn.py \
    --device cuda:0 \
    --custom_dist shubert \
    --beta 2.0 \
    --save_dir ./test

# With w_ratio (scalarized multi-objective)
python train_gfn.py \
    --device cuda:0 \
    --custom_dist mix2 \
    --w_ratio 0.5,0.5 \
    --save_dir ./test
