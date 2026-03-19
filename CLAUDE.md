# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MUR (Momentum Uncertainty Guided Reasoning) is a research implementation for optimizing LLM reasoning chains. It selectively triggers expensive candidate sampling/verification only when model confidence drops below a threshold, reducing computation by ~50% while improving accuracy. Paper: arXiv:2507.14958.

## Running Experiments

```bash
# Using shell scripts (recommended, paths pre-configured)
bash scripts/run_guided_search_mur.sh
bash scripts/run_guided_search_pre_calibration.sh

# Direct invocation
python guided_search-mur.py --policy <model_path> --critic <critic_path> --data_path data/gpqa_diamond_test.json
python guided_search-pre_calibration.py --policy <model_path> --critic <critic_path> --data_path data/gpqa_diamond_test.json
python guided_search-per_step_scale.py --policy <model_path> --critic <critic_path> --data_path data/gpqa_diamond_test.json
```

Key arguments: `--scaling_rate` (default 0.8-0.9, controls trigger sensitivity), `--momentum_rate` (0.9, MUR only), `--max_steps` (20), `--candidate_num` (4), `--verify_num` (1, number of critic evaluations per candidate), `--aim_gpu` (GPU device ID).

## Evaluation

```bash
python eval/math_verifier.py --test_data_path <results_json> --verifier <verifier_model_path>
python eval/eval_gpqa_cot.py  # pattern-based answer extraction
```

Results are saved to `res/`, timing/token stats to `res/time/`.

## Architecture

**Three guided search variants**, all using an external critic model to score candidates:

| File | Trigger Condition | Description |
|---|---|---|
| `guided_search-mur.py` | `exp(cur_signal) < exp(momentum_uncertainty) * scaling_rate` | Momentum-based: triggers when step confidence drops below EMA baseline |
| `guided_search-pre_calibration.py` | `cur_signal < calibration_mean_logp * scaling_rate` | Pre-calibration: generates a full trajectory first without TTS to compute mean uncertainty, then uses that as threshold |
| `guided_search-per_step_scale.py` | `if True:` (always) | Baseline: triggers candidate sampling at every step |

**Common pipeline** (when triggered):
1. Generate `candidate_num` alternative steps
2. Critic model evaluates each candidate via analyze → judge flow
3. Select candidate with highest "Yes" token logprob
4. Replace current step with best candidate

**Output data**: All scripts save `step_uncertainties` per question (step-level `avg_logp`, threshold info, whether triggered). Pre-calibration additionally saves `calibration_mean_logp` and `calibration_step_logps`.

**Shared utilities**: `utils/generate_prompts.py` provides critic prompt templates (`ciritique_last_generation_math` for MATH/AIME, `ciritique_last_generation` for GPQA).

## Dependencies

Requires vLLM for inference. Install with: `pip install -r requirements.txt`

## Datasets

Four test datasets in `data/`: `math_500_test.json`, `aime2024_test.json`, `aime2025_test.json`, `gpqa_diamond_test.json`.
