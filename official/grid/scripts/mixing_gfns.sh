#!/bin/bash
# Evaluate mixing GFlowNets (one example per mixing type)

# Scalarization (simplex sweep)
python eval_mixing.py \
    --model_saved_path_list pretrained_models/shubert.pt,pretrained_models/diagonal.pt \
    --custom_dist mix2 \
    --mixing_type scalarization \
    --n_reward_fns 2 \
    --result_saved_path ./test/mixing_scalarization.png \
    --device cuda:0

# Harmonic mean
python eval_mixing.py \
    --model_saved_path_list pretrained_models/shubert.pt,pretrained_models/diagonal.pt \
    --custom_dist harmonic_mean_shu_diag \
    --mixing_type harmonic_mean \
    --result_saved_path ./test/mixing_hm.png \
    --device cuda:0

# Contrast (towards model 1)
python eval_mixing.py \
    --model_saved_path_list pretrained_models/shubert.pt,pretrained_models/diagonal.pt \
    --custom_dist contrast_shu_diag \
    --mixing_type contrast \
    --result_saved_path ./test/mixing_contrast.png \
    --device cuda:0

# Scalarization without u (simplex sweep)
python eval_mixing.py \
    --model_saved_path_list pretrained_models/shubert.pt,pretrained_models/diagonal.pt \
    --custom_dist mix2 \
    --mixing_type scalarization_without_u \
    --n_reward_fns 2 \
    --result_saved_path ./test/mixing_scalarization_without_u.png \
    --device cuda:0

# Harmonic mean without u
python eval_mixing.py \
    --model_saved_path_list pretrained_models/shubert.pt,pretrained_models/diagonal.pt \
    --custom_dist harmonic_mean_shu_diag \
    --mixing_type harmonic_mean_without_u \
    --result_saved_path ./test/mixing_hm_without_u.png \
    --device cuda:0

# Contrast without u
python eval_mixing.py \
    --model_saved_path_list pretrained_models/shubert.pt,pretrained_models/diagonal.pt \
    --custom_dist contrast_shu_diag \
    --mixing_type contrast_without_u \
    --result_saved_path ./test/mixing_contrast_without_u.png \
    --device cuda:0
