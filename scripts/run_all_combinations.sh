#!/bin/bash
# Run all TTS method and trigger combinations on all datasets.

set -e

POLICY_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/Qwen3-8B"
CRITIC_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/genprm1.5B"

POLICY_URL="http://localhost:8000/v1"
CRITIC_URL="http://localhost:8001/v1"

WORKERS=64
NUM_ROLLOUTS=8
MAX_STEPS=20
CANDIDATE_NUM=4
VERIFY_NUM=1
MOMENTUM_RATE=0.9
SCALING_RATE=0.9
CLUSTER_NUM=2

DATASETS=(
    "data/aime2024_test.json"
    "data/aime2025_test.json"
    "data/gpqa_diamond_test.json"
    "data/math_500_test.json"
)

for DATA_PATH in "${DATASETS[@]}"; do
    for METHOD in guided_search llm_as_a_critic phi_decoding; do
        for TRIGGER in mur per_step; do
            echo "Running $METHOD with $TRIGGER on $DATA_PATH..."

            CMD="python tts_experiment.py \
                --tts_method $METHOD --trigger $TRIGGER \
                --policy_url $POLICY_URL --policy_model_name $POLICY_PATH \
                --data_path $DATA_PATH --workers $WORKERS \
                --max_steps $MAX_STEPS --candidate_num $CANDIDATE_NUM \
                --verify_num $VERIFY_NUM --momentum_rate $MOMENTUM_RATE \
                --scaling_rate $SCALING_RATE --cluster_num $CLUSTER_NUM \
                --num_rollouts $NUM_ROLLOUTS"

            [ "$METHOD" != "phi_decoding" ] && CMD="$CMD --critic_url $CRITIC_URL --critic_model_name $CRITIC_PATH"

            eval $CMD || echo "Failed: $METHOD-$TRIGGER on $DATA_PATH"
        done
    done
done
