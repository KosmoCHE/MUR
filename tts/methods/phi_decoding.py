"""Phi Decoding TTS method.

Generates N candidate steps, then performs foresight reranking using
TF-IDF + KMeans clustering. Only uses the policy model (no critic needed).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from tts.client import VLLMClient
from tts.prompts import build_policy_prompt
from tts.methods import TTSResult


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def apply_tts(policy_client: VLLMClient, critic_client,
              policy_tokenizer, critic_tokenizer,
              question: str, current_traj: list[str], step_idx: int,
              args, policy_stop_token: str, critic_stop_token: str) -> TTSResult:
    """Generate candidates and select the best via foresight reranking."""
    policy_tokens = 0

    # 1. Generate N candidate steps
    prompt = build_policy_prompt(
        policy_tokenizer, question, current_traj[:-1], step_idx, policy_stop_token)
    candidates_results = policy_client.complete(
        prompt, max_tokens=2048, temperature=0.6,
        stop=["Step"], logprobs=1, n=args.candidate_num)

    candidates = [r.text.strip() for r in candidates_results]
    all_logps = [r.avg_logp for r in candidates_results]
    policy_tokens += sum(r.num_tokens for r in candidates_results)

    # 2. Foresight: generate one more step from each candidate
    foresight_inputs = []
    for cand in candidates:
        foresight_prompt = prompt + cand
        foresight_inputs.append(foresight_prompt)

    foresight_results = policy_client.complete_batch(
        foresight_inputs, max_tokens=2048, temperature=0.6,
        stop=["Step"], logprobs=1)
    policy_tokens += sum(r.num_tokens for results in foresight_results for r in results)

    # Collect foresight texts and scores (filter empty)
    foresight_texts = []
    foresight_scores = []
    valid_indices = []
    for i, results in enumerate(foresight_results):
        text = results[0].text.strip() if results else ""
        if text:
            foresight_texts.append(text)
            foresight_scores.append(results[0].avg_logp)
            valid_indices.append(i)

    if not foresight_texts:
        # Fallback: pick by candidate logprob
        best_idx = int(np.argmax(all_logps))
        return TTSResult(
            best_step_text=f"Step{step_idx}: {candidates[best_idx]}",
            best_avg_logp=all_logps[best_idx],
            policy_tokens=policy_tokens,
            critic_tokens=0,
            metadata={'step_idx': str(step_idx), 'selected_idx': str(best_idx),
                      'candidates': candidates, 'fallback': 'empty_foresight'}
        )

    # 3. TF-IDF + KMeans clustering
    cluster_num = min(args.cluster_num, len(foresight_texts))
    try:
        X = TfidfVectorizer().fit_transform(foresight_texts)
        kmeans = KMeans(n_clusters=cluster_num, n_init='auto').fit(X)
        labels = kmeans.labels_

        cluster_sizes = [list(labels).count(i) for i in labels]
        cluster_probs = softmax(cluster_sizes)
        foresight_probs = softmax(foresight_scores)
        combined_probs = [(foresight_probs[i] + cluster_probs[i]) / 2
                          for i in range(len(foresight_scores))]

        best_foresight_idx = np.random.choice(
            range(len(foresight_texts)), p=combined_probs)
    except Exception:
        # Clustering failed, fallback to foresight logprob sampling
        fallback_probs = softmax(foresight_scores)
        best_foresight_idx = np.random.choice(
            range(len(foresight_scores)), p=fallback_probs)

    # Map back to original candidate index
    best_idx = valid_indices[best_foresight_idx]

    return TTSResult(
        best_step_text=f"Step{step_idx}: {candidates[best_idx]}",
        best_avg_logp=all_logps[best_idx],
        policy_tokens=policy_tokens,
        critic_tokens=0,
        metadata={
            'step_idx': str(step_idx),
            'selected_idx': str(best_idx),
            'candidates': candidates,
        }
    )
