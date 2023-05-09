from typing import Union, Tuple, Optional
from dataclasses import dataclass

from cuco import config_parser, Config

from .Frequency import Frequency
from .FrequencyUnit import FrequencyUnit


@config_parser(module_name='base.eval')
@dataclass
class ExperimentEvalConfig(Config):
    frequency: Union[int, Tuple[int, str]] = Frequency(8, FrequencyUnit.EPISODE)
    frequency_max: Optional[Union[int, Tuple[int, str]]] = None
    start: int = 16
    n_episodes: int = 1
    max_episode_length: int = None
    run_during_training: bool = True
    step_delay: float = 0.2
