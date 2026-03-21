"""Guided Search TTS method.

Generates N candidate steps via policy, scores each with critic
(analyze -> judge -> extract "Yes" logprob), selects the best candidate.
"""
import numpy as np

from tts.client import VLLMClient
from tts.prompts import build_policy_prompt, build_critic_analyze_prompt_for_candidates
from tts.methods import TTSResult


def apply_tts(policy_client: VLLMClient, critic_client: VLLMClient,
              policy_tokenizer, critic_tokenizer,
              question: str, current_traj: list[str], step_idx: int,
              args, policy_stop_token: str, critic_stop_token: str) -> TTSResult:
    """Generate candidates and select the best one using critic scoring."""
    policy_tokens = 0
    critic_tokens = 0

    # 1. Generate candidate_num alternative steps
    prompt = build_policy_prompt(
        policy_tokenizer, question, current_traj[:-1], step_idx, policy_stop_token)
    candidates_results = policy_client.complete(
        prompt, max_tokens=2048, temperature=0.6,
        stop=["Step"], logprobs=1, n=args.candidate_num)

    candidates = [r.text.strip() for r in candidates_results]
    logps = [r.avg_logp for r in candidates_results]
    policy_tokens += sum(r.num_tokens for r in candidates_results)

    # 2. Build critic analyze prompts for each candidate
    analyze_inputs = build_critic_analyze_prompt_for_candidates(
        critic_tokenizer, question, current_traj[:-1], candidates,
        step_idx, critic_stop_token, args.data_path)

    # 3. Critic analyze phase (batch)
    analyze_results = critic_client.complete_batch(
        analyze_inputs, max_tokens=1024, temperature=0.6,
        stop=['</analyze>\n', '```python'], n=args.verify_num,
        extra_body={"include_stop_str_in_output": True})
    critic_tokens += sum(r.num_tokens for results in analyze_results for r in results)

    # 4. Build judge prompts
    output_start = "<output>\n**Judgement**: $\\boxed"
    output_inputs = []
    for idx, results in enumerate(analyze_results):
        for result in results:
            analyze_text = result.text.strip()
            output_inputs.append(analyze_inputs[idx] + analyze_text + output_start)

    # 5. Critic judge phase (batch)
    judge_results = critic_client.complete_batch(
        output_inputs, max_tokens=2048, temperature=0.6,
        stop=['</output>\n', '</think>\n', '```python'],
        logprobs=1,
        extra_body={"include_stop_str_in_output": True})
    critic_tokens += sum(r.num_tokens for results in judge_results for r in results)

    # 6. Extract "Yes" token logprob for each candidate
    yes_logps = [0.0] * len(candidates)
    for idx, results in enumerate(judge_results):
        for result in results:
            for i, token_str in enumerate(result.tokens):
                if token_str == 'Yes':
                    yes_logps[idx // args.verify_num] += np.exp(result.token_logprobs[i])
                    break

    best_idx = int(np.argmax(yes_logps))

    return TTSResult(
        best_step_text=f"Step{step_idx}: {candidates[best_idx]}",
        best_avg_logp=logps[best_idx],
        policy_tokens=policy_tokens,
        critic_tokens=critic_tokens,
        metadata={
            'step_idx': str(step_idx),
            'selected_idx': str(best_idx),
            'candidates': candidates,
            'yes_scores': yes_logps,
        }
    )
