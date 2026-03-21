"""LLM-as-a-Critic TTS method.

Critic evaluates the current step (not candidates). If the critic judges
the step as incorrect, the policy model revises it using critic feedback.
"""
import re

from tts.client import VLLMClient
from tts.prompts import build_critic_analyze_prompt, build_revision_prompt
from tts.methods import TTSResult


def extract_critic_judgment(analyze_text: str, output_text: str):
    """Extract and clean critic judgment from analyze/output text."""
    critic_full = f"<analyze>{analyze_text}<output>{output_text}"

    try:
        analyze_content = re.search(
            r'<analyze>(.*?)</analyze>', critic_full, re.DOTALL).group(1)
    except AttributeError:
        analyze_content = analyze_text

    try:
        output_content = re.search(
            r'<output>(.*?)</(?:output|think)>', critic_full, re.DOTALL).group(1)
    except AttributeError:
        output_content = output_text

    # Clean up analyze text for use in revision prompt
    analyze_content = (analyze_content
                       .replace('<analyze>', '').replace('</analyze>', '')
                       .replace('paragraph_', 'Step').replace('paragraph', 'Step')
                       .replace('**Judgement**:', 'So the correctness of the step is:'))

    return analyze_content, output_content


def apply_tts(policy_client: VLLMClient, critic_client: VLLMClient,
              policy_tokenizer, critic_tokenizer,
              question: str, current_traj: list[str], step_idx: int,
              args, policy_stop_token: str, critic_stop_token: str) -> TTSResult:
    """Evaluate current step with critic; revise if judged incorrect."""
    policy_tokens = 0
    critic_tokens = 0

    # 1. Build critic analyze prompt for the current step
    analyze_input = build_critic_analyze_prompt(
        critic_tokenizer, question, current_traj,
        step_idx, critic_stop_token, args.data_path)

    # 2. Critic analyze phase
    analyze_results = critic_client.complete(
        analyze_input, max_tokens=4096, temperature=0.6,
        stop=['</analyze>\n', '```python'],
        extra_body={"include_stop_str_in_output": True})
    analyze_text = analyze_results[0].text.strip()
    critic_tokens += analyze_results[0].num_tokens

    # 3. Critic judge phase
    output_start = "<output>\n**Judgement**: $\\boxed"
    judge_input = analyze_input + analyze_text + output_start
    judge_results = critic_client.complete(
        judge_input, max_tokens=4096, temperature=0.6,
        stop=['</output>\n', '</think>\n', '```python'],
        extra_body={"include_stop_str_in_output": True})
    output_text = judge_results[0].text.strip()
    critic_tokens += judge_results[0].num_tokens

    # 4. Extract judgment
    analyze_content, judge_content = extract_critic_judgment(analyze_text, output_text)

    # 5. If incorrect, revise with policy
    if 'yes' not in judge_content.lower():
        revision_prompt = build_revision_prompt(
            policy_tokenizer, question, current_traj,
            step_idx, analyze_content, policy_stop_token)

        revision_results = policy_client.complete(
            revision_prompt, max_tokens=2048, temperature=0.6,
            stop=["Step"], logprobs=1)
        revised_text = revision_results[0].text.strip()
        policy_tokens += revision_results[0].num_tokens

        return TTSResult(
            best_step_text=f"Step{step_idx}: {revised_text}",
            best_avg_logp=revision_results[0].avg_logp,
            policy_tokens=policy_tokens,
            critic_tokens=critic_tokens,
            metadata={
                'step_idx': str(step_idx),
                'judgment': 'No',
                'revised': True,
                'analyze_content': analyze_content,
            }
        )

    # Step was judged correct, keep as-is
    return TTSResult(
        best_step_text=current_traj[-1],
        best_avg_logp=None,  # No new logprob; caller should keep cur_signal
        policy_tokens=policy_tokens,
        critic_tokens=critic_tokens,
        metadata={
            'step_idx': str(step_idx),
            'judgment': 'Yes',
            'revised': False,
        }
    )
