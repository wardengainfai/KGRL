from enum import Enum
from typing import Optional, Callable


class LearningRateSchedule(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    QUARTIC = "quartic"
    DEC = "dec"

    def build_generator(self, initial_value: float, final_value: Optional[float], **kwargs) -> Callable[[float], float]:
        if self == LearningRateSchedule.CONSTANT:
            return lambda remaining_progress: initial_value
        elif self == LearningRateSchedule.LINEAR:
            return linear_schedule(initial_value=initial_value, final_value=final_value, **kwargs)

        elif self == LearningRateSchedule.QUADRATIC:
            if final_value:
                return lambda x: (initial_value-final_value) * pow(x, 2) + final_value
            return lambda remaining_progress: pow(remaining_progress, 2) * initial_value
        elif self == LearningRateSchedule.CUBIC:
            if final_value:
                return lambda x: (initial_value-final_value) * pow(x, 3) + final_value
            return lambda remaining_progress: pow(remaining_progress, 3) * initial_value
        elif self == LearningRateSchedule.QUARTIC:
            if final_value:
                return lambda x: (initial_value-final_value) * pow(x, 4) + final_value
            return lambda remaining_progress: pow(remaining_progress, 4) * initial_value
        elif self == LearningRateSchedule.DEC:
            if final_value:
                return lambda x: (initial_value-final_value) * pow(x, 10) + final_value
            return lambda remaining_progress: pow(remaining_progress, 10) * initial_value


def linear_schedule(
        initial_value: float,
        final_value: Optional[float] = None,
        truncate: bool = True,
        **kwargs,
) -> Callable[[float], float]:
    if final_value is not None and truncate:
        def func(progress_remaining: float) -> float:
            return max(progress_remaining * initial_value, final_value)

        return func
    elif final_value and not truncate:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value + (
                        1 - progress_remaining) * final_value

        return func
    else:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func