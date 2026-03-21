#!/bin/bash

# Evaluate results with automatic dataset type detection
# - gpqa_diamond → eval_gpqa_cot.py (pattern-based answer extraction)
# - math_500 / aime2024 / aime2025 → math_verifier.py (general verifier model)
#
# Usage:
#   bash scripts/run_eval.sh                        # evaluate all JSON/JSONL files in res/
#   bash scripts/run_eval.sh res/some_result.jsonl   # evaluate a specific file
#   bash scripts/run_eval.sh res/some_result.jsonl 1 # specify GPU id (for math/aime)

VERIFIER_PATH="/root/siton-data-7b4916f6b64e457fa4ce0508dc033853/yh/general_verifier"
GPU_ID=${2:-0}

if [ -n "$1" ]; then
    FILES=("$1")
else
    FILES=(res/*.json res/*.jsonl)
fi

for FILE in "${FILES[@]}"; do
    # Skip glob patterns that didn't match any files
    [ -f "$FILE" ] || continue

    BASENAME=$(basename "$FILE")
    # Strip both .json and .jsonl extensions
    EVAL_NAME="${BASENAME%.jsonl}"
    EVAL_NAME="${EVAL_NAME%.json}"
    EVAL_OUT="res/eval/${EVAL_NAME}.txt"

    if [ -f "$EVAL_OUT" ]; then
        echo "[SKIP] Already evaluated: $FILE → $EVAL_OUT"
        continue
    fi

    echo "=========================================="
    echo "Evaluating: $FILE"

    if [[ "$BASENAME" == gpqa* ]]; then
        echo "Dataset: GPQA → eval_gpqa_cot.py"
        echo "=========================================="
        python eval/eval_gpqa_cot.py --test_file "$FILE"
    elif [[ "$BASENAME" == math* ]] || [[ "$BASENAME" == aime* ]]; then
        echo "Dataset: MATH/AIME → math_verifier.py"
        echo "=========================================="
        python eval/math_verifier.py --test_file "$FILE" --verifier "$VERIFIER_PATH" --aim_gpu "$GPU_ID"
    else
        echo "Unknown dataset type, skipping: $BASENAME"
        echo "=========================================="
    fi
    echo ""
done
