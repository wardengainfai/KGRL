from dataclasses import dataclass

from typing import Union, List, Optional, Mapping, Any
from cuco import config_parser

from ..base.ExperimentConfig import ExperimentConfig
from .KGMinigridWrapperConfig import KGMinigridWrapperConfig


@config_parser(module_name='minigrid.experiment')
@dataclass
class MinigridExperimentConfig(ExperimentConfig):

    experiment_env_name: str = "kg-minigrid-env-v0",
    minigrid_env_name: str = "MiniGrid-DoorKey-8x8-v0",
    minigrid_fixed_seed: Union[bool, int, List[int]] = 42,
    max_env_steps: int = 20000,
    kg_wrapper_config: Optional[Union[KGMinigridWrapperConfig, Mapping[str, Any]]] = KGMinigridWrapperConfig(),

    @classmethod
    def load(
        cls,
        experiment_env_name: str = "kg-minigrid-env-v0",
        minigrid_env_name: str = "MiniGrid-DoorKey-8x8-v0",
        minigrid_fixed_seed: Union[bool, int, List[int]] = 42,
        kg_wrapper_config: Optional[Union[KGMinigridWrapperConfig, Mapping[str, Any]]] = KGMinigridWrapperConfig(),
        **kwargs
    ):
        return cls(
            minigrid_env_name=minigrid_env_name,
            minigrid_fixed_seed=minigrid_fixed_seed,
            **super(cls, cls).load(**kwargs, experiment_env_name = experiment_env_name, kg_wrapper_config = kg_wrapper_config).__dict__
        )
