from dataclasses import dataclass
import openai


@dataclass
class CompletionResult:
    """Normalized result from a vLLM server completion."""
    text: str
    tokens: list[str]
    token_logprobs: list[float]
    cumulative_logprob: float
    avg_logp: float
    num_tokens: int


class VLLMClient:
    """Wrapper around the OpenAI-compatible vLLM server API."""

    def __init__(self, base_url: str, model: str):
        self.client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model

    def _parse_choice(self, choice) -> CompletionResult:
        text = choice.text
        if choice.logprobs and choice.logprobs.token_logprobs:
            token_logprobs = [lp if lp is not None else 0.0
                              for lp in choice.logprobs.token_logprobs]
            tokens = list(choice.logprobs.tokens)
        else:
            token_logprobs = []
            tokens = []

        cumulative_logprob = sum(token_logprobs)
        num_tokens = len(tokens)
        avg_logp = cumulative_logprob / (num_tokens + 1e-8)

        return CompletionResult(
            text=text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            cumulative_logprob=cumulative_logprob,
            avg_logp=avg_logp,
            num_tokens=num_tokens,
        )

    def complete(self, prompt: str, max_tokens: int, temperature: float,
                 stop: list[str] | None = None, n: int = 1,
                 logprobs: int | None = None,
                 extra_body: dict | None = None) -> list[CompletionResult]:
        """Complete a single prompt, returning n results."""
        kwargs = dict(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        if stop is not None:
            kwargs['stop'] = stop
        if logprobs is not None:
            kwargs['logprobs'] = logprobs
        if extra_body is not None:
            kwargs['extra_body'] = extra_body

        response = self.client.completions.create(**kwargs)
        return [self._parse_choice(c) for c in response.choices]

    def complete_batch(self, prompts: list[str], max_tokens: int,
                       temperature: float, stop: list[str] | None = None,
                       n: int = 1, logprobs: int | None = None,
                       extra_body: dict | None = None) -> list[list[CompletionResult]]:
        """Complete multiple prompts in one request. Returns a list of lists,
        one inner list per prompt, each containing n CompletionResults."""
        if not prompts:
            return []

        kwargs = dict(
            model=self.model,
            prompt=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        if stop is not None:
            kwargs['stop'] = stop
        if logprobs is not None:
            kwargs['logprobs'] = logprobs
        if extra_body is not None:
            kwargs['extra_body'] = extra_body

        response = self.client.completions.create(**kwargs)

        # Group choices by prompt index
        results: list[list[CompletionResult]] = [[] for _ in prompts]
        for choice in response.choices:
            prompt_idx = choice.index // n
            results[prompt_idx].append(self._parse_choice(choice))

        return results
