from utils.generate_prompts import ciritique_last_generation, ciritique_last_generation_math


def get_system_prompt(data_path: str) -> str:
    """Determine system prompt based on dataset name."""
    path = data_path.lower()
    if 'math' in path:
        return ''
    elif 'aime' in path:
        return 'You are a helpful math assistant. '
    elif any(x in path for x in ['reclor', 'gpqa', 'logiqa']):
        return ('You are a helpful assistant. Here is a question and four candidate answers. '
                'You need to reason step by step and choose the most likely answer from the '
                'four candidate answers. Answer "A", "B", "C", or "D".')
    elif 'strategyqa' in path:
        return ('You are a helpful assistant. After each step, you may receive a feedback from '
                'the user, indicating that the previous step is incorrect. You should then revise '
                'your solution accordingly. Please answer "Yes" or "No".')
    return ''


def build_policy_prompt(tokenizer, question: str, traj: list[str],
                        step_idx: int, stop_token: str) -> str:
    """Build policy model input prompt with trajectory so far."""
    chat = [{'role': 'user',
             'content': f"Q: {question}\nAlways end your solution with the phrase "
                        f"'the answer is' followed by your final answer. "
                        f"Start your solution with 'Step{step_idx}:'\n"}]
    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    input_text = input_text.replace(stop_token, "").strip()
    if step_idx > 0:
        input_text += '\n'.join(traj) + f'\nStep{step_idx}:'
    else:
        input_text += '\nStep0:'
    return input_text


def build_critic_analyze_prompt(tokenizer, question: str, traj: list[str],
                                step_idx: int, stop_token: str,
                                data_path: str) -> str:
    """Build critic input up to the <analyze> prefix."""
    if 'math' in data_path.lower() or 'aime' in data_path.lower():
        critic_prompt_dict = ciritique_last_generation_math(question, traj)
    else:
        critic_prompt_dict = ciritique_last_generation(question, traj)

    critic_input = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': critic_prompt_dict['system_prompt']},
         {'role': 'user', 'content': critic_prompt_dict['user_prompt']}],
        tokenize=False,
        add_generation_prompt=True
    )
    analyze_start = f"<analyze>\nLet's analyze the paragraph {step_idx} step by step: "
    return critic_input.replace(stop_token, "").strip() + analyze_start


def build_critic_analyze_prompt_for_candidates(tokenizer, question: str, traj: list[str],
                                               candidates: list[str], step_idx: int,
                                               stop_token: str, data_path: str) -> list[str]:
    """Build critic analyze prompts for each candidate step."""
    analyze_inputs = []
    for candidate in candidates:
        if 'math' in data_path.lower() or 'aime' in data_path.lower():
            critic_prompt_dict = ciritique_last_generation_math(
                question, traj + [candidate])
        else:
            critic_prompt_dict = ciritique_last_generation(
                question, traj + [candidate])

        critic_input = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': critic_prompt_dict['system_prompt']},
             {'role': 'user', 'content': critic_prompt_dict['user_prompt']}],
            tokenize=False,
            add_generation_prompt=True
        )
        analyze_start = f"<analyze>\nLet's analyze the paragraph {step_idx} step by step: "
        analyze_inputs.append(critic_input.replace(stop_token, "").strip() + analyze_start)
    return analyze_inputs


def build_revision_prompt(tokenizer, question: str, traj: list[str],
                          step_idx: int, analyze_content: str,
                          stop_token: str) -> str:
    """Build multi-turn revision prompt for llm_as_a_critic method."""
    messages = [
        {'role': 'user',
         'content': f"Q: {question}\nAlways end your solution with the phrase "
                    f"'the answer is' followed by your final answer. "
                    f"Start your solution with 'Step{step_idx}:'\n"},
        {'role': 'assistant', 'content': '\n'.join(traj)},
        {'role': 'user',
         'content': f"\nYour previous solution is incorrect.\n{analyze_content}\n"
                    f"Please revise your solution."},
        {'role': 'assistant', 'content': f'Refined Step{step_idx}: '}
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False)
    return input_text.replace(stop_token, "").strip()
