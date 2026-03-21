import numpy as np


class TriggerState:
    """Mutable state maintained across steps for trigger decisions."""

    def __init__(self):
        self.momentum_uncertainty: float = 0.0
        self.step_count: int = 0


class MURTrigger:
    """Momentum-based trigger: fires when step confidence drops below EMA baseline.

    Condition: exp(cur_signal) < exp(momentum_uncertainty) * scaling_rate, and step > 0.
    """

    def __init__(self, scaling_rate: float, momentum_rate: float):
        self.scaling_rate = scaling_rate
        self.momentum_rate = momentum_rate

    def should_trigger(self, cur_signal: float, state: TriggerState) -> bool:
        if state.step_count == 0:
            return False
        return np.exp(cur_signal) < np.exp(state.momentum_uncertainty) * self.scaling_rate

    def update(self, cur_signal: float, state: TriggerState):
        state.momentum_uncertainty = (self.momentum_rate * state.momentum_uncertainty
                                      + (1 - self.momentum_rate) * cur_signal)
        state.step_count += 1

    def get_threshold(self, state: TriggerState) -> float:
        if state.momentum_uncertainty == 0:
            return 0.0
        return float(np.log(np.exp(state.momentum_uncertainty) * self.scaling_rate))


class PerStepTrigger:
    """Always-trigger baseline: fires at every step."""

    def __init__(self, momentum_rate: float = 0.9):
        self.momentum_rate = momentum_rate

    def should_trigger(self, cur_signal: float, state: TriggerState) -> bool:
        return True

    def update(self, cur_signal: float, state: TriggerState):
        state.momentum_uncertainty = (self.momentum_rate * state.momentum_uncertainty
                                      + (1 - self.momentum_rate) * cur_signal)
        state.step_count += 1

    def get_threshold(self, state: TriggerState) -> float:
        return 0.0
