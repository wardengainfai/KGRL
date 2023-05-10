from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass

import numpy as np
from rdflib import URIRef

import gym

from ....config import KGMinigridWrapperConfig
from ....base.environments.kg_wrapper import KGWrapper, KGObservation, GraphHandler

from ..kg.minigrid_graph_generator import MinigridGraphGenerator
from ..utils import improve_encoding
from kgrl.base.kg.nlu.er_linking import er_linker_emb


@dataclass
class MinigridKGObservation(KGObservation):
    """
    Observations of the wrapped minigrid environment.
    """
    #state = position of the agent in the grid
    image: Optional[np.ndarray] = None  # view of the agent
    direction: Optional[int] = None  # 3: "North", 0: "East", 1: "South", 2
    mission: Optional[str] = None  # string describing the mission to receive reward
    mission_embeddings: Optional[Tuple[np.ndarray, np.ndarray]] = None  # triples embeddings and its confidence scores

class MinigridKGWrapper(KGWrapper[dict, np.ndarray, MinigridKGObservation]):
    def __init__(
            self,
            env,
            config: Optional[KGMinigridWrapperConfig] = None,
            graph_generator: Optional[MinigridGraphGenerator] = None,
    ):
        if config is None:
            config = KGMinigridWrapperConfig()

        if graph_generator is None:
            graph_generator = MinigridGraphGenerator()

        super().__init__(env, config, graph_generator)

        self.mission_embedding: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def get_env_representation(self) -> np.ndarray:
        return improve_encoding(self.env.grid.encode())

    def observation_to_kg_entity(self, observation: dict) -> URIRef:
        # the plain observation in minigrid consists of the view of the agent
        # along with the direction and mission string
        position = self.env.agent_pos
        return self.graph_generator.get_cell_instance(position)

    def observation_to_kg_observation(self, observation: dict) -> MinigridKGObservation:
        return MinigridKGObservation(
            state=self.env.agent_pos,
            image=observation["image"] if "image" in self.config.original_observation_keys else None,
            direction=observation["direction"] if "direction" in self.config.original_observation_keys else None,
            mission=observation["mission"] if "mission" in self.config.original_observation_keys else None,
        )

    def transform_mission(self, mission: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the mission to a list of embedded (n_triples)-triples.
        """
        mission_embedding = er_linker_emb(
            kg=self.graph_handler.graph,
            mission=mission,
            kg_emb=self.graph_handler.embedding_results,
            n_triples=self.config.n_triples,
            padding=self.config.er_padding,
        )
        return mission_embedding

    def get_observation_space(self) -> gym.spaces.Space:
        """
        Return the augmented observation space.
        """
        obs_space_dict: gym.space.Dict = super().get_observation_space()

        # note that the image part of the observation corresponds to the view of
        # the agent encoded as array of triples with (object_idx, color_idx, state)
        # the original implementation messes with some image Policies (e.g. with
        # the `MultiInputPolicy` for `DQN` in SB3.
        new_space_dict = {
            "state": gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.env.width, self.env.height]),
                dtype="uint8"
            ),
            "image": gym.spaces.Box(
                low=0,
                high=15,
                shape=(self.agent_view_size, self.agent_view_size, 3),
                dtype='uint8'
            ),
            "direction": gym.spaces.Discrete(4),
            "mission": None,  # todo: add a observation space for the sting mission. if the
                              #  mission is the same for all steps its not needed.
        }
        if self.config.compute_embeddings:
            new_space_dict['mission_embeddings'] = gym.spaces.Tuple((
                gym.spaces.Box(
                    low=np.array(
                        [[np.amin(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)]
                        +[np.amin(self.graph_handler.embedding_model.relation_representations[0]().detach().cpu().numpy() * 1.01, axis=0)]
                        +[np.amin(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)]
                        for _ in range(self.config.n_triples)]).reshape((self.config.n_triples, 3, self.config.model.dim)),

                    high=np.array(
                        [[np.amax(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)]
                        +[np.amax(self.graph_handler.embedding_model.relation_representations[0]().detach().cpu().numpy() * 1.01, axis=0)]
                        +[np.amax(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)]
                        for _ in range(self.config.n_triples)]).reshape((self.config.n_triples, 3, self.config.model.dim)),

                    shape=(self.config.n_triples, 3, self.config.model.dim)
                ),
                gym.spaces.Box(low=np.zeros(self.config.n_triples), high=np.ones(self.config.n_triples))
            ))

        for key, value in new_space_dict.items():
            if value is not None and key in self.config.original_observation_keys:
                obs_space_dict[key] = value
        return obs_space_dict

    def observation(self, observation: Dict, as_dict: bool = True) -> Union[KGObservation, Dict[str, np.ndarray]]:
        mission = observation['mission']
        observation: MinigridKGObservation = super().observation(observation=observation, as_dict=False)
        if self.config.transform_mission:
            if self.mission_embedding is None:
                self.mission_embedding = self.transform_mission(mission)
            observation.mission_embeddings = self.mission_embedding

        if self.config.observation_as_dict and as_dict:
            return observation.to_dict()
        return observations

