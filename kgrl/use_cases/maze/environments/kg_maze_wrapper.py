from typing import Optional

import numpy as np
from rdflib import Graph, URIRef

from ....base.environments.kg_wrapper import KGObservation, KGWrapper
from ....config import KGMazeWrapperConfig

from ..utils.mazeutils import gymmaze2pyamaze
from ..kg.maze_graph_generator import MazeGraphGenerator, MazeRepresentation
from ..utils.MazeMap import MazeMap, MazeRoom


class KGMazeObservation(KGObservation):
    """
    Class for the new observations of the maze env wrapped with KGMazeWrapper
    """
    state: np.ndarray


class KGMazeWrapper(KGWrapper[np.ndarray, MazeRepresentation, KGMazeObservation]):
    """
    Class to wrap the Gym-Maze env to allow use of KGs. This wrapper is supposed
    to work with the gym_maze environments
    """

    def __init__(
            self,
            env,
            config: Optional[KGMazeWrapperConfig] = None,
            graph_generator: Optional[MazeGraphGenerator] = None
    ):
        if config is None:
            config = KGMazeWrapperConfig()

        if graph_generator is None:
            graph_generator = MazeGraphGenerator()
        super().__init__(env, config, graph_generator)

    def get_env_representation(self) -> MazeRepresentation:
        gym_map = self.env.maze_view.maze.maze_cells
        return gymmaze2pyamaze(gym_map)  # need a pyamaze-format map

    def get_environment_kg(self) -> Graph:
        """Get rdf representation of the maze and return rdf-lib graph"""
        # for a static environment we only need to generate the graph once
        gym_map = self.env.maze_view.maze.maze_cells
        self.graph_generator.generate_graph(MazeMap(gym_map), include_walls = self.config.model.include_walls)  # need a pyamaze-format map
        return self.graph_generator.domain_kg

    def observation_to_kg_entity(self, observation: np.ndarray) -> URIRef:
        return MazeRoom(*tuple(observation.astype(int))).uri

    def observation_to_kg_observation(self, observation: np.ndarray) -> KGMazeObservation:
        # return KGMazeObservation(state=torch.tensor(observation.astype(int), device = 'cuda'))
        return KGMazeObservation(state=observation.astype(int))
