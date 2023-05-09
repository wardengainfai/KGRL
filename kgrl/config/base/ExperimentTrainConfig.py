from typing import Union, Tuple, Optional
from dataclasses import dataclass

from cuco import config_parser, Config

from .Frequency import Frequency
from .FrequencyUnit import FrequencyUnit
from .BatchMode import BatchMode
from .LearningRate import LearningRate


@config_parser(module_name='base.train')
@dataclass
class ExperimentTrainConfig(Config):
    # trainer_config: dict = None not needed without rllib

    steps: int = 100  # gradient steps in sb3
    total_timesteps: int = 1000000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    frequency: Union[int, Tuple[int, str]] = Frequency(4, FrequencyUnit.STEP)
    buffer_size: int = 200000
    max_episodes: int = 2000
    observation_type: str = 'dict'  # either dict or array

    rollout_fragment_length: int = 1000
    batch_mode: BatchMode = BatchMode.COMPLETE_EPISODES
    batch_size: int = 128
    gamma: float = 0.99  # discount factor
    target_update_interval: int = 10000
    learning_starts: int = 10000

    num_gpus: int = 1
    num_workers: int = 0

    evaluation_interval: int = None

    learning_rate: LearningRate = None

    @property
    def trainer(self):
        config = self.trainer_config

        config["framework"] = "torch"
        config["rollout_fragment_length"] = self.rollout_fragment_length
        config["batch_mode"] = self.batch_mode
        config["train_batch_size"] = self.batch_size
        config["num_gpus"] = self.num_gpus
        config["num_workers"] = self.num_workers

        if self.evaluation_interval is not None:
            config['evaluation_interval'] = self.evaluation_interval

        return config
