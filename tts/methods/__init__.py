from dataclasses import dataclass, field


@dataclass
class TTSResult:
    """Result from a TTS method application."""
    best_step_text: str
    best_avg_logp: float | None  # None if no new logprob (e.g. critic judged correct)
    policy_tokens: int
    critic_tokens: int
    metadata: dict = field(default_factory=dict)
