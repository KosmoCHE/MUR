#!/bin/bash

# Guided Search with MUR (Momentum-based Uncertainty Reduction)
# Policy: Qwen3-8B
# Critic: genprm1.5B

POLICY_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/Qwen3-8B"
CRITIC_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/genprm1.5B"

python guided_search-mur.py \
    --policy "$POLICY_PATH" \
    --critic "$CRITIC_PATH" \
    --data_path data/gpqa_diamond_test.json \
    --max_steps 20 \
    --candidate_num 4 \
    --verify_num 1 \
    --momentum_rate 0.9 \
    --scaling_rate 0.9 \
    --gpus 1 \
    --aim_gpu 0
