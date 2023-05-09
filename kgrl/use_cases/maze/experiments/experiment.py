"""classes and utils for running experiments
"""
import os

import logging
from typing import Dict, Any, Optional, Union, List

import numpy as np

import gym_maze.envs.maze_view_2d
from gym_maze.envs.maze_env import MazeEnv

from ....base.experiments.experiment import Experimenter
from ....config import MazeExperimentConfig, StateMazeWrapperConfig

from ..environments.kg_maze_wrapper import KGMazeWrapper, KGMazeWrapperConfig
from ..environments.gym_maze_wrapper import GymMazeStatesWrapper


class MazeExperimenter(Experimenter):

    def __init__(
            self,
            config: Optional[MazeExperimentConfig] = None,
            **kwargs,
    ):
        # see https://docs.ray.io/en/latest/rllib-training.html#common-parameters
        if not isinstance(config, MazeExperimentConfig):
            if config is not None:
                logging.warning("Provided config has to be of `MazeExperimentConfig` class.")
            logging.info("Experiment Config set to default values.")
            config = MazeExperimentConfig()

        super().__init__(
            config=config,
            **kwargs,
        )

    def _create_env_force_headless(self, config: Optional[Any] = None) -> Union[KGMazeWrapper, GymMazeStatesWrapper, MazeEnv]:
        return self._create_env(config, True)
        # env = self._create_env(env_config, False)
        # env.start_evaluation() 
        # return env

    def _create_env(self, config: Optional[Any] = None, force_headless: bool = False) -> Union[KGMazeWrapper, GymMazeStatesWrapper, MazeEnv]:
        # print(self.config.experiment_headless, force_headless)
        """Environment creator function for rllib trainer"""
        base_env = MazeEnv(  # If knowledge graph is enabled this env is wrapped into a kg environment (see below)
            maze_file=self.config.maze_file,  # maze file has precedence
            maze_size=self.config.maze_size,
            mode=None,
            enable_render=False if force_headless else not self.config.experiment_headless
        )

        if self.config.maze_max_episode_steps:
            # if we have a step limit wrap the env in TimeLimit
            from gym.wrappers.time_limit import TimeLimit
            base_env = TimeLimit(base_env, max_episode_steps=self.config.maze_max_episode_steps)

        if self.config.kg_wrapper_enabled:
            config = config if isinstance(config, KGMazeWrapperConfig) else self.config.kg_wrapper_config
            return KGMazeWrapper(base_env, config)
        elif self.config.states_wrapper_enabled:
            config = config if isinstance(config, StateMazeWrapperConfig) else self.config.states_wrapper_config
            return GymMazeStatesWrapper(base_env, config)
        else:
            return base_env

    def reset_experiment(
            self,
            checkpoint: str = None,
            generate_maze: bool = False,
            **kwargs
    ) -> None:
        """Clean the Experimenter to start from scratch or from checkpoint."""
        super().reset_experiment(checkpoint=checkpoint)

        # store the path to the saved maze so that the created envs have the same maze
        if generate_maze or self.config.maze_file is None:
            self.config.maze_file = self.save_maze(maze=self.generate_maze())

    def generate_maze(self) -> np.ndarray:
        return gym_maze.envs.maze_view_2d.Maze(
            maze_size=self.config.maze_size,
            has_loops=False,
            num_portals=0,
        ).maze_cells

    def save_maze(self, maze: np.ndarray, path: str = None) -> str:
        if path is None:
            path = os.path.join(self.config.checkpoint_directory, "maze.npy")
        with open(path, "wb+") as file:
            # noinspection PyTypeChecker
            # np.save also accepts file-objects see https://numpy.org/doc/stable/reference/generated/numpy.save.html?highlight=save#numpy.save
            np.save(file, maze)
        return path

    def evaluate_checkpoint(self, checkpoint) -> Dict[str, Any]:
        """
        todo: Evaluate the policy in a checkpoint or list of checkpoints.
        """
        new_config = self.agent.config
        new_config["evaluation_interval"] = 1
        self.agent.reset(new_config)
        self.agent.restore(checkpoint)
        # return self.agent.evaluate(,  # TODO: Statement is not complete

    def run_trial(self, n_training_runs: int = 20, envs: Optional[Union[str, List[str]]] = None, log_name_suffix: str = "", **kwargs):
        if envs is None:
            envs = ["kg", "plain", "states"]
        experiment_dir = self.config.checkpoint_directory
        self._register_env()
        self.config.log_dir = os.path.join(self.config.checkpoint_directory, "tb_logs")
        for i in range(n_training_runs):
            self.config.checkpoint_directory = os.path.join(experiment_dir, str(i))
            if not os.path.isdir(self.config.checkpoint_directory):
                os.makedirs(self.config.checkpoint_directory)
            self.config.kg_wrapper_config.embedding_checkpoint_dir = os.path.join(experiment_dir, str(i), "graph_embedding")
            maze_path = os.path.join(self.config.checkpoint_directory, "maze.npy")
            self.config.maze_file = maze_path \
                if os.path.isfile(maze_path)\
                else self.save_maze(maze=self.generate_maze(), path=maze_path,)
            for environment in envs:
                self.config.kg_wrapper_enabled = False
                self.config.states_wrapper_enabled = False

                if environment == "kg":
                    self.config.kg_wrapper_enabled = True
                elif environment == "states":
                    self.config.states_wrapper_enabled = True
                elif environment == "plain":
                    pass
                # self._register_env(reregister=True)
                self.reset_experiment()
                print("running training number {} for {} in environment {}".format(i, self.config.experiment_name, environment))
                self.run_training(tb_log_name=self.config.experiment_name+"_"+environment+"_"+log_name_suffix+"_"+str(i))
                print("done Training.")

    def evaluate_trial(self, path=None):
        pass
