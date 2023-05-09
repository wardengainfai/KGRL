from typing import Optional
from dataclasses import dataclass

from cuco import config_parser, Config

from .LearningRateSchedule import LearningRateSchedule


@config_parser(module_name='base.learning-rate')
@dataclass
class LearningRate(Config):
    initial_value: float
    final_value: Optional[float] = None
    schedule: LearningRateSchedule = LearningRateSchedule.CONSTANT
    kwargs: dict = None

    @property
    def as_callable(self):
        return self.schedule.build_generator(self.initial_value, self.final_value)

    @staticmethod
    def load(initial_value: float, final_value: Optional[float] = None, schedule: str = 'constant', kwargs: dict = None):
        return LearningRate(initial_value=initial_value, final_value=final_value, schedule=LearningRateSchedule(schedule), kwargs=kwargs)
