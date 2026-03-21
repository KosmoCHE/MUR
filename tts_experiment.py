"""Unified TTS experiment script.

Supports multiple TTS methods (guided_search, phi_decoding, llm_as_a_critic)
and trigger strategies (mur, per_step). Uses vLLM server API for inference.

Usage:
    # Start vLLM servers first (see scripts/start_servers.sh)
    python tts_experiment.py \
        --tts_method guided_search --trigger mur \
        --policy_url http://localhost:8000/v1 \
        --critic_url http://localhost:8001/v1 \
        --policy_model_name <model_path> \
        --critic_model_name <critic_path> \
        --data_path data/gpqa_diamond_test.json
"""
import os
import json
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from transformers import AutoTokenizer

from tts.client import VLLMClient
from tts.prompts import build_policy_prompt
from tts.triggers import MURTrigger, PerStepTrigger, TriggerState


def parse_args():
    parser = argparse.ArgumentParser(description='Unified TTS Experiment')

    # TTS method and trigger
    parser.add_argument('--tts_method',
                        choices=['guided_search', 'phi_decoding', 'llm_as_a_critic'],
                        required=True, help='TTS method to use')
    parser.add_argument('--trigger', choices=['mur', 'per_step'], default='mur',
                        help='Trigger strategy (default: mur)')

    # Server endpoints
    parser.add_argument('--policy_url', type=str, default='http://localhost:8000/v1',
                        help='vLLM server URL for policy model')
    parser.add_argument('--critic_url', type=str, default='http://localhost:8001/v1',
                        help='vLLM server URL for critic model')
    parser.add_argument('--policy_model_name', type=str, required=True,
                        help='Policy model name/path (for tokenizer and server)')
    parser.add_argument('--critic_model_name', type=str, default=None,
                        help='Critic model name/path (not needed for phi_decoding)')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data JSON')

    # Shared hyperparameters
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--candidate_num', type=int, default=4)
    parser.add_argument('--verify_num', type=int, default=1)
    parser.add_argument('--scaling_rate', type=float, default=0.9)
    parser.add_argument('--momentum_rate', type=float, default=0.9)

    # Phi decoding specific
    parser.add_argument('--cluster_num', type=int, default=2,
                        help='Number of clusters for phi_decoding foresight reranking')

    # Concurrency
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of concurrent workers for processing questions')

    # Output
    parser.add_argument('--file_name', type=str, default=None,
                        help='Output file name (auto-generated if not specified)')

    args = parser.parse_args()

    # Validate: guided_search and llm_as_a_critic need critic
    if args.tts_method != 'phi_decoding' and args.critic_model_name is None:
        parser.error('--critic_model_name is required for guided_search and llm_as_a_critic')

    # Auto-generate file name
    if args.file_name is None:
        dataset = os.path.basename(args.data_path).replace('.json', '').replace('_test', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.file_name = f'{dataset}-{args.tts_method}-{args.trigger}-{timestamp}'

    return args


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, tokenizer.eos_token


def process_question(idx, example, policy_client, critic_client,
                     policy_tokenizer, critic_tokenizer,
                     policy_stop_token, critic_stop_token,
                     trigger, tts_method, args):
    """Process a single question through the TTS pipeline."""
    question = example['input']
    current_traj = []
    candidate_traj = []
    step_uncertainties = []
    trigger_state = TriggerState()
    get_answer = False
    all_policy_tokens = 0
    all_critic_tokens = 0

    step_idx = 0
    for step_idx in range(args.max_steps):
        try:
            # 1. Generate next step
            input_text = build_policy_prompt(
                policy_tokenizer, question, current_traj, step_idx, policy_stop_token)
            results = policy_client.complete(
                input_text, max_tokens=2048, temperature=0.6,
                stop=["Step"], logprobs=1)
            output = results[0]
            all_policy_tokens += output.num_tokens

            cur_signal = output.avg_logp
            current_traj.append(f"Step{step_idx}: {output.text.strip()}")

            # 2. Check trigger
            triggered = trigger.should_trigger(cur_signal, trigger_state)

            # 3. Record step uncertainty
            step_uncertainties.append({
                'step_idx': step_idx,
                'avg_logp': float(cur_signal),
                'momentum_uncertainty': float(trigger_state.momentum_uncertainty),
                'threshold': trigger.get_threshold(trigger_state),
                'triggered': bool(triggered),
                'step_text': f"Step{step_idx}: {output.text.strip()}"
            })

            # 4. Apply TTS if triggered
            if triggered:
                tts_result = tts_method.apply_tts(
                    policy_client, critic_client,
                    policy_tokenizer, critic_tokenizer,
                    question, current_traj, step_idx, args,
                    policy_stop_token, critic_stop_token)

                current_traj[-1] = tts_result.best_step_text
                # For llm_as_a_critic: if step was judged correct, keep cur_signal
                if tts_result.best_avg_logp is not None:
                    cur_signal = tts_result.best_avg_logp
                all_policy_tokens += tts_result.policy_tokens
                all_critic_tokens += tts_result.critic_tokens

                candidate_traj.append({
                    'step_idx': str(step_idx),
                    'step_uncertainty': str(np.exp(-cur_signal)),
                    'momentum_uncertainty/gamma': str(
                        np.exp(-trigger_state.momentum_uncertainty) / args.momentum_rate),
                    'selected_idx': tts_result.metadata.get('selected_idx', '0'),
                    'candidates': tts_result.metadata.get('candidates', []),
                    'original_traj': current_traj[-1],
                    **{k: v for k, v in tts_result.metadata.items()
                       if k not in ('selected_idx', 'candidates')}
                })

            # 5. Update trigger state
            trigger.update(cur_signal, trigger_state)

            # 6. Check for answer
            if "the answer is" in ''.join(current_traj).lower():
                get_answer = True
                break

        except Exception as e:
            print(f"[Q{idx}] Step {step_idx} error: {e}")
            continue

    # Final fallback if no answer found
    if not get_answer:
        try:
            input_text = build_policy_prompt(
                policy_tokenizer, question, current_traj, step_idx, policy_stop_token)
            results = policy_client.complete(
                input_text, max_tokens=8096, temperature=0.6, logprobs=1)
            current_traj.append(results[0].text.strip())
            all_policy_tokens += results[0].num_tokens
        except Exception as e:
            print(f"[Q{idx}] Final fallback error: {e}")

    return {
        'question': question,
        'ground_truth': example['target'],
        'current_traj': '\n'.join(current_traj),
        'final_answer': current_traj[-1] if current_traj else 'No answer',
        'candidate_traj': candidate_traj,
        'step_uncertainties': step_uncertainties,
    }, all_policy_tokens, all_critic_tokens


def main():
    args = parse_args()

    # Load tokenizers
    policy_tokenizer, policy_stop_token = load_tokenizer(args.policy_model_name)
    critic_tokenizer, critic_stop_token = None, None
    if args.critic_model_name:
        critic_tokenizer, critic_stop_token = load_tokenizer(args.critic_model_name)

    # Create server clients
    policy_client = VLLMClient(base_url=args.policy_url, model=args.policy_model_name)
    critic_client = None
    if args.critic_model_name:
        critic_client = VLLMClient(base_url=args.critic_url, model=args.critic_model_name)

    # Create trigger
    if args.trigger == 'mur':
        trigger = MURTrigger(args.scaling_rate, args.momentum_rate)
    else:
        trigger = PerStepTrigger(args.momentum_rate)

    # Load TTS method
    if args.tts_method == 'guided_search':
        from tts.methods import guided_search as tts_method
    elif args.tts_method == 'phi_decoding':
        from tts.methods import phi_decoding as tts_method
    elif args.tts_method == 'llm_as_a_critic':
        from tts.methods import llm_as_a_critic as tts_method

    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    os.makedirs('res', exist_ok=True)
    os.makedirs('res/time', exist_ok=True)

    all_res = [None] * len(test_data)
    total_policy_tokens = 0
    total_critic_tokens = 0
    completed_count = 0
    write_lock = Lock()
    start_time = time.time()

    def save_results():
        """Save current results to disk (thread-safe)."""
        with write_lock:
            # Collect non-None results in order
            results_to_save = [r for r in all_res if r is not None]
            with open(f'res/{args.file_name}.json', 'w') as f:
                json.dump(results_to_save, f, indent=4)

    if args.workers <= 1:
        # Sequential processing
        for idx, example in enumerate(test_data):
            print(f"Processing {idx} / {len(test_data)}")
            result, p_tokens, c_tokens = process_question(
                idx, example, policy_client, critic_client,
                policy_tokenizer, critic_tokenizer,
                policy_stop_token, critic_stop_token,
                trigger, tts_method, args)
            all_res[idx] = result
            total_policy_tokens += p_tokens
            total_critic_tokens += c_tokens
            save_results()
    else:
        # Concurrent processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for idx, example in enumerate(test_data):
                future = executor.submit(
                    process_question,
                    idx, example, policy_client, critic_client,
                    policy_tokenizer, critic_tokenizer,
                    policy_stop_token, critic_stop_token,
                    trigger, tts_method, args)
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result, p_tokens, c_tokens = future.result()
                    all_res[idx] = result
                    total_policy_tokens += p_tokens
                    total_critic_tokens += c_tokens
                    completed_count += 1
                    print(f"Completed {completed_count} / {len(test_data)} (Q{idx})")
                    save_results()
                except Exception as e:
                    print(f"[Q{idx}] Failed: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time taken: {elapsed:.2f} seconds")

    # Save timing stats
    with open(f'res/time/{args.file_name}.txt', 'w') as f:
        f.write(f'\n\n{args.file_name}  time: {elapsed:.2f}\n\n')
        f.write(f'all_policy_output_tokens: {total_policy_tokens}\n')
        f.write(f'all_critic_output_tokens: {total_critic_tokens}\n')
        f.write(f'tts_method: {args.tts_method}\n')
        f.write(f'trigger: {args.trigger}\n')
        f.write(f'workers: {args.workers}\n')


if __name__ == '__main__':
    main()
