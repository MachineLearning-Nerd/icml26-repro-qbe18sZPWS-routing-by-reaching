#!/bin/bash
# Train 2-way classifier (one example)

python train_classifier_2way.py \
    --model_path_1 pretrained_models/base-gfns/shubert.pt \
    --model_path_2 pretrained_models/base-gfns/sphere.pt \
    --custom_dist_1 shubert \
    --custom_dist_2 sphere \
    --save_dir ./test \
    --device cuda:0

# Evaluate
python eval_classifier_2way.py \
    --model_path_1 pretrained_models/base-gfns/shubert.pt \
    --model_path_2 pretrained_models/base-gfns/sphere.pt \
    --cls_path pretrained_models/classifiers/classifier_shubert_sphere.pt \
    --custom_dist_1 shubert \
    --custom_dist_2 sphere \
    --result_saved_path ./test/eval_cls_shubert_sphere.png \
    --device cuda:0
