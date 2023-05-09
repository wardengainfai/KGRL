from typing import Tuple, Optional, List

import numpy as np

import gym
from gym_maze.envs.maze_env import MazeEnv

from ....config import StateMazeWrapperConfig, RevealState


class GymMazeStatesWrapper(gym.ObservationWrapper):
    """Augment the observed states with additional neighbors"""
    def __init__(
            self,
            env: MazeEnv,
            config: Optional[StateMazeWrapperConfig] = None
    ):
        super().__init__(env)
        self.config = config if config is not None else StateMazeWrapperConfig(reveal = RevealState.N_HOP)
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self):
        if self.config.reveal_n_hop_neighbors or self.config.reveal_neighbors:
            low = np.zeros((self.config.truncate, 2), dtype=int)
            high = np.concatenate((
                np.full((self.config.truncate, 1), self.env.maze_size[0]),
                np.full((self.config.truncate, 1), self.env.maze_size[1])
            ), 1, dtype=int)
            return gym.spaces.Box(low, high, dtype=np.int64)
        else:
            return self.observation_space

    def get_neighbors(self, state: Tuple[int, int]) -> List[np.ndarray]:
        """Return all direct neighbors of state excluding the state"""
        wall_dict = self.env.maze_view.maze.get_walls_status(
            self.env.maze_view.maze.maze_cells[state]
        )
        neighbors = []
        for direction, passage in wall_dict.items():
            if passage:
                neighbors.append(np.array(state) + self.env.maze_view.maze.COMPASS[direction])
        return neighbors

    def get_n_neighbors(
            self,
            state: np.ndarray,
    ) -> np.ndarray:
        """get the neighboring cells"""
        stack = [state]
        neighbors = []
        while stack and len(neighbors) < self.config.n_neighbors:
            current = tuple(stack.pop())  # Tuple
            if current not in neighbors:
                neighbors.append(current)  # List[Tuple[int, int]]
            else:
                continue
            stack += self.get_neighbors(current)
        observation = np.array(neighbors, dtype=int)
        if self.config.verbose:
            print('State: ')
            print(state)
            print('Observation: ')
            print(observation)
        return observation

    def get_n_hop(
            self,
            state,
    ):
        """Get the cells that are n_hop transitions away"""
        stack = [state]
        neighbors = []
        num_hop = 0
        while num_hop <= self.config.n_hop:
            new_stack = []
            for s in stack:
                s = tuple(s)
                if s not in neighbors:
                    neighbors.append(s)
                new_stack += self.get_neighbors(s)
            stack = new_stack
            num_hop += 1
        if self.config.truncate is not None:
            neighbors = neighbors[:self.config.truncate] \
                if len(neighbors) > self.config.truncate\
                else neighbors + [np.full((2,), -1) for _ in range(self.config.truncate - len(neighbors))]
        return np.array(neighbors, dtype=int)

    def observation(self, state: np.ndarray) -> np.ndarray:
        if self.config.reveal_n_hop_neighbors:
            return self.get_n_hop(state)
        elif self.config.reveal_neighbors:
            return self.get_n_neighbors(state)
        else:
            return state.astype(int)

    def render(self, mode="human", **kwargs):
        if self.config.render_visible:
            state = self.env.state
            # we need a list of cells
            visible_cells = [np.array(cell) for cell in self.observation(state).tolist()]
            return self.env.render(mode, cells=visible_cells, **kwargs)
        return self.env.render(mode, **kwargs)
