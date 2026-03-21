#!/bin/bash
# Run TTS experiment using vLLM server backend.
# Make sure servers are started first (see scripts/start_servers.sh).
#
# Usage:
#   bash scripts/run_experiment.sh

# ---- Configuration ----
POLICY_MODEL="/path/to/Qwen3-8B"
CRITIC_MODEL="/path/to/genprm1.5B"
DATA_PATH="data/aime2025_test.json"

TTS_METHOD="guided_search"   # guided_search | phi_decoding | llm_as_a_critic
TRIGGER="mur"                # mur | per_step

POLICY_URL="http://localhost:8000/v1"
CRITIC_URL="http://localhost:8001/v1"

WORKERS=1
NUM_ROLLOUTS=1
MAX_STEPS=20
CANDIDATE_NUM=4
VERIFY_NUM=1
MOMENTUM_RATE=0.9
SCALING_RATE=0.9
CLUSTER_NUM=2  # phi_decoding only
# ------------------------

CMD="python tts_experiment.py \
    --tts_method $TTS_METHOD \
    --trigger $TRIGGER \
    --policy_url $POLICY_URL \
    --critic_url $CRITIC_URL \
    --policy_model_name $POLICY_MODEL \
    --data_path $DATA_PATH \
    --max_steps $MAX_STEPS \
    --candidate_num $CANDIDATE_NUM \
    --verify_num $VERIFY_NUM \
    --momentum_rate $MOMENTUM_RATE \
    --scaling_rate $SCALING_RATE \
    --cluster_num $CLUSTER_NUM \
    --workers $WORKERS \
    --num_rollouts $NUM_ROLLOUTS"

# Add critic model if not phi_decoding
if [ "$TTS_METHOD" != "phi_decoding" ]; then
    CMD="$CMD --critic_model_name $CRITIC_MODEL"
fi

echo "Running: $CMD"
eval $CMD
