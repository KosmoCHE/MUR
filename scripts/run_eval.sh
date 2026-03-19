#!/bin/bash

# Evaluate results with automatic dataset type detection
# - gpqa_diamond → eval_gpqa_cot.py (pattern-based answer extraction)
# - math_500 / aime2024 / aime2025 → math_verifier.py (general verifier model)
#
# Usage:
#   bash scripts/run_eval.sh                        # evaluate all JSON files in res/
#   bash scripts/run_eval.sh res/some_result.json    # evaluate a specific file
#   bash scripts/run_eval.sh res/some_result.json 1  # specify GPU id (for math/aime)

GPU_ID=${2:-0}

if [ -n "$1" ]; then
    FILES=("$1")
else
    FILES=(res/*.json)
fi

for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    echo "=========================================="
    echo "Evaluating: $FILE"

    if [[ "$BASENAME" == gpqa* ]]; then
        echo "Dataset: GPQA → eval_gpqa_cot.py"
        echo "=========================================="
        python eval/eval_gpqa_cot.py --test_file "$FILE"
    elif [[ "$BASENAME" == math* ]] || [[ "$BASENAME" == aime* ]]; then
        echo "Dataset: MATH/AIME → math_verifier.py"
        echo "=========================================="
        python eval/math_verifier.py --test_file "$FILE" --aim_gpu "$GPU_ID"
    else
        echo "Unknown dataset type, skipping: $BASENAME"
        echo "=========================================="
    fi
    echo ""
done
