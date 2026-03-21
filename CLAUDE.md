# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MUR (Momentum Uncertainty Guided Reasoning) is a research implementation for optimizing LLM reasoning chains. It selectively triggers expensive candidate sampling/verification only when model confidence drops below a threshold, reducing computation by ~50% while improving accuracy. Paper: arXiv:2507.14958.

## Running Experiments

Experiments use a vLLM server backend (OpenAI-compatible API) for high-throughput inference.

### 1. Start vLLM servers

```bash
# Start policy + critic servers (customize model paths, GPU, ports)
bash scripts/start_servers.sh /path/to/Qwen3-8B /path/to/genprm1.5B

# For phi_decoding (no critic needed), omit critic path
bash scripts/start_servers.sh /path/to/Qwen3-8B

# Environment variables: POLICY_PORT, CRITIC_PORT, POLICY_GPU, CRITIC_GPU, POLICY_MEM, CRITIC_MEM
```

### 2. Run experiment

```bash
# Using shell script (edit scripts/run_experiment.sh to configure)
bash scripts/run_experiment.sh

# Direct invocation
python tts_experiment.py \
    --tts_method guided_search \
    --trigger mur \
    --policy_url http://localhost:8000/v1 \
    --critic_url http://localhost:8001/v1 \
    --policy_model_name /path/to/Qwen3-8B \
    --critic_model_name /path/to/genprm1.5B \
    --data_path data/gpqa_diamond_test.json \
    --workers 4
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--tts_method` | (required) | `guided_search`, `phi_decoding`, or `llm_as_a_critic` |
| `--trigger` | `mur` | `mur` (momentum-based) or `per_step` (always trigger) |
| `--scaling_rate` | 0.9 | Trigger sensitivity |
| `--momentum_rate` | 0.9 | EMA rate for momentum uncertainty |
| `--max_steps` | 20 | Maximum reasoning steps |
| `--candidate_num` | 4 | Candidates per trigger (guided_search, phi_decoding) |
| `--verify_num` | 1 | Critic evaluations per candidate (guided_search) |
| `--cluster_num` | 2 | KMeans clusters (phi_decoding only) |
| `--workers` | 1 | Concurrent question processing threads |

## Evaluation

```bash
python eval/math_verifier.py --test_data_path <results_json> --verifier <verifier_model_path>
python eval/eval_gpqa_cot.py  # pattern-based answer extraction
bash scripts/run_eval.sh <results_json>  # auto-detects dataset type
```

Results are saved to `res/`, timing/token stats to `res/time/`.

## Architecture

### TTS Methods (`tts/methods/`)

| Method | File | Description | Needs Critic |
|---|---|---|---|
| Guided Search | `tts/methods/guided_search.py` | Generate N candidates, critic scores each via analyze→judge, select by "Yes" logprob | Yes |
| Phi Decoding | `tts/methods/phi_decoding.py` | Generate N candidates, foresight reranking with TF-IDF + KMeans clustering | No |
| LLM-as-a-Critic | `tts/methods/llm_as_a_critic.py` | Critic evaluates current step, policy revises if judged incorrect | Yes |

### Trigger Strategies (`tts/triggers.py`)

| Trigger | Condition |
|---|---|
| MUR | `exp(cur_signal) < exp(momentum_uncertainty) * scaling_rate`, step > 0 |
| Per-Step | Always trigger (baseline) |

### Package Structure

```
tts/
  client.py        # vLLM server client wrapper (OpenAI-compatible API)
  prompts.py       # Prompt construction utilities
  triggers.py      # Trigger strategies (MUR, per_step)
  methods/
    guided_search.py
    phi_decoding.py
    llm_as_a_critic.py
tts_experiment.py  # Unified entry script
```

**Shared utilities**: `utils/generate_prompts.py` provides critic prompt templates (`ciritique_last_generation_math` for MATH/AIME, `ciritique_last_generation` for GPQA).

## Dependencies

Requires vLLM (server mode), openai, scikit-learn. Install with: `pip install -r requirements.txt`

## Datasets

Four test datasets in `data/`: `math_500_test.json`, `aime2024_test.json`, `aime2025_test.json`, `gpqa_diamond_test.json`.
