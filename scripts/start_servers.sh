#!/bin/bash
# Start vLLM OpenAI-compatible API servers for policy and critic models.
#
# Usage:
#   bash scripts/start_servers.sh <policy_model_path> [critic_model_path]
#   # For phi_decoding (no critic needed), omit critic_model_path.
#
# Environment variables:
#   POLICY_PORT  - Policy server port (default: 8000)
#   CRITIC_PORT  - Critic server port (default: 8001)
#   POLICY_GPU   - GPU ids for policy (default: 0,1)
#   CRITIC_GPU   - GPU ids for critic (default: 0,1)
#   POLICY_MEM   - GPU memory utilization for policy (default: 0.6)
#   CRITIC_MEM   - GPU memory utilization for critic (default: 0.45)
#   POLICY_DP    - Data parallel size for policy (default: 2)
#   CRITIC_DP    - Data parallel size for critic (default: 2)

set -e

POLICY_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/Qwen3-8B"
CRITIC_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/genprm1.5B"

POLICY_PORT="${POLICY_PORT:-8000}"
CRITIC_PORT="${CRITIC_PORT:-8001}"
POLICY_GPU="${POLICY_GPU:-0,1}"
CRITIC_GPU="${CRITIC_GPU:-0,1}"
POLICY_MEM="${POLICY_MEM:-0.7}"
CRITIC_MEM="${CRITIC_MEM:-0.2}"
POLICY_DP="${POLICY_DP:-2}"
CRITIC_DP="${CRITIC_DP:-2}"

echo "Starting policy server on port $POLICY_PORT (GPU $POLICY_GPU, dp=$POLICY_DP)..."
CUDA_VISIBLE_DEVICES=$POLICY_GPU vllm serve "$POLICY_PATH" \
    --port "$POLICY_PORT" \
    --gpu-memory-utilization "$POLICY_MEM" \
    --data-parallel-size "$POLICY_DP" \
    --max-model-len 16192 \
    --trust-remote-code &
POLICY_PID=$!
echo "Policy server PID: $POLICY_PID"

echo "Waiting for policy server to be ready..."
MAX_WAIT=300  # 最长等待 300 秒
ELAPSED=0
until curl -s http://localhost:$POLICY_PORT/health > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Policy server failed to start within ${MAX_WAIT}s"
        kill $POLICY_PID 2>/dev/null
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  Still waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
done
echo "Policy server ready after ${ELAPSED}s."

if [ -n "$CRITIC_PATH" ]; then
    echo "Starting critic server on port $CRITIC_PORT (GPU $CRITIC_GPU, dp=$CRITIC_DP)..."
    CUDA_VISIBLE_DEVICES=$CRITIC_GPU vllm serve "$CRITIC_PATH" \
        --port "$CRITIC_PORT" \
        --gpu-memory-utilization "$CRITIC_MEM" \
        --data-parallel-size "$CRITIC_DP" \
        --max-model-len 16192 \
        --trust-remote-code &
    CRITIC_PID=$!
    echo "Critic server PID: $CRITIC_PID"
fi

echo ""
echo "Servers started. To stop: kill $POLICY_PID ${CRITIC_PID:-}"
echo "Waiting for servers..."
wait
