from typing import Optional, Tuple, Union, Mapping, Any
from cuco import config_parser
from ..base.ExperimentConfig import ExperimentConfig
from ..base.Connectivity import Connectivity
from .StateMazeWrapperConfig import StateMazeWrapperConfig


@config_parser(module_name = 'maze.experiment')
class MazeExperimentConfig(ExperimentConfig):
    r"""
    Class for configuring Experiments with KG mazes.
    """

    def __init__(
        self,
        experiment_env_name: str = 'kg-maze-env-v0',
        maze_file: Optional[str] = None,
        maze_size: Tuple[int, int] = None,
        maze_connectivity: Connectivity = Connectivity.DEFAULT,
        maze_max_episode_steps: Union[int, bool] = True,
        states_wrapper_enabled: bool = False,
        states_wrapper_config: Optional[StateMazeWrapperConfig] = None,
        **kwargs
    ):
        super().__init__(
            experiment_env_name = experiment_env_name,
            **kwargs
        )

        if states_wrapper_enabled and self.kg_wrapper_enabled:
            raise ValueError("Choose either states wrapper or KG wrapper.")

        self.maze_connectivity = maze_connectivity
        self.maze_size = maze_size
        self.maze_file = maze_file
        self.maze_max_episode_steps = maze_max_episode_steps
        self.states_wrapper_enabled = states_wrapper_enabled
        self.states_wrapper_config = states_wrapper_config

    @classmethod
    def load(
        cls,
        experiment_env_name: str = 'kg-maze-env-v0',
        maze_file: Optional[str] = None,
        maze_size: Tuple[int, int] = None,
        maze_connectivity: str = Connectivity.DEFAULT.value,
        maze_max_episode_steps: Union[int, bool] = True,
        maze_max_episode_factor: int = 400,
        states_wrapper_enabled: bool = False,
        states_wrapper_config: Optional[Union[StateMazeWrapperConfig, Mapping[str, Any]]] = StateMazeWrapperConfig(),
        **kwargs
    ):
        if maze_size is None and maze_file is None:
            maze_file = "maze2d_5x5.npy"
            maze_size = (5, 5)

        if maze_max_episode_steps == 1:
            maze_max_episode_steps = maze_size[0] * maze_size[1] * maze_max_episode_factor

        # print(states_wrapper_enabled, states_wrapper_config)

        # if states_wrapper_enabled:
        #     if states_wrapper_config is None:
        #         states_wrapper_config = GymMazeStatesWrapperConfig(reveal = RevealState.NEIGHBORS)
        #     elif isinstance(states_wrapper_config, dict):
        #         states_wrapper_config = GymMazeStatesWrapperConfig(**states_wrapper_config)

        return MazeExperimentConfig(
            maze_file = maze_file,
            maze_size = maze_size,
            maze_connectivity = Connectivity(maze_connectivity),
            maze_max_episode_steps = maze_max_episode_steps,
            states_wrapper_enabled = states_wrapper_enabled,
            states_wrapper_config = states_wrapper_config,
            **super(cls, cls).load(**kwargs, experiment_env_name = experiment_env_name).__dict__
        )

    def dump(self) -> dict:
        raise NotImplementedError('Coming soon')
