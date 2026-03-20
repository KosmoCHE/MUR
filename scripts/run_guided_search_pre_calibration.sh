#!/bin/bash

# Guided Search with Pre-Calibration
# Policy: Qwen3-8B
# Critic: genprm1.5B

POLICY_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/Qwen3-8B"
CRITIC_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/genprm1.5B"
DATA_PATH="data/aime2025_test.json"
python guided_search-pre_calibration.py \
    --policy "$POLICY_PATH" \
    --critic "$CRITIC_PATH" \
    --data_path "$DATA_PATH" \
    --max_steps 20 \
    --candidate_num 4 \
    --verify_num 1 \
    --scaling_rate 0.9 \
    --gpus 1 \
    --aim_gpu 1
