from typing import Optional, Union, Any, List
import logging

import gym
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper, FlatObsWrapper, DictObservationSpaceWrapper

from ..environments.minigrid_wrappers import NoImageWrapper
from ....config import KGMinigridWrapperConfig
from ....config import MinigridExperimentConfig

from ....base.experiments.experiment import Experimenter

from ..environments.minigrid_kg_wrapper import MinigridKGWrapper


class MinigridExperimenter(Experimenter):
    def __init__(
            self,
            config: Optional[MinigridExperimentConfig] = None,
            **kwargs,
    ):
        if not isinstance(config, MinigridExperimentConfig):
            if config is not None:
                logging.warning("Provided config has to be of `MinigridExperimentConfig` class.")
            logging.info("Experiment Config set to default values.")
            config = MinigridExperimentConfig()

        super().__init__(
            config=config,
            **kwargs,
        )
        self.config = config

    def _create_env(self, config: MinigridExperimentConfig = None, force_headless: bool = False) -> Union[gym.Env]:
        """Provide the environment."""
        if config is None:
            config = self.config

        env = gym.make(config.minigrid_env_name)  # max_steps = self.config.minigrid_max_steps)
        # set the max steps to the desired value
        env.unwrapped.max_steps = config.max_env_steps
        if config.minigrid_fixed_seed:
            # we can select a fixed seed for the env with the reseedwrapper
            env = ReseedWrapper(
                env,
                seeds=[config.minigrid_fixed_seed] if not isinstance(config.minigrid_fixed_seed, list) else config.minigrid_fixed_seed,
            )
        if config.kg_wrapper_enabled:
            config = config if isinstance(config, KGMinigridWrapperConfig) else config.kg_wrapper_config
            return MinigridKGWrapper(env, config)

        env = DictObservationSpaceWrapper(env)
        # make sure the 'image' observation space is not interpreted as an rgb image
        env = NoImageWrapper(env)
        return env

    def _create_env_force_headless(self, config: MinigridExperimentConfig = None) -> Union[gym.Env]:
        return self._create_env(config, True)

    def run_trial(
        self,
        n_training_runs: int = 20,
        envs: Optional[Union[str, List[str]]] = None,
        log_name_suffix: str = "",
        **kwargs
    ) -> None:
        pass

    def evaluate_trial(self, path: str = None) -> None:
        pass

    def _set_study_name(self, study_name: str, study_name_suffix: str = ''):
        if study_name is None:
            study_name = self.config.minigrid_env_name
        self.study_name = study_name + study_name_suffix
        return self.study_name
