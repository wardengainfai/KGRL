import os
import shutil

from abc import ABC, abstractmethod
from typing import Union, Any, Optional, Generic, TypeVar, Tuple, Dict
from dataclasses import dataclass, asdict

import gym
import numpy as np
import pykeen.nn
import torch
from rdflib import Graph, URIRef

from pykeen.triples import TriplesFactory
from pykeen.pipeline import PipelineResult
from pykeen.models import Model, ERModel
from pykeen.training.training_loop import CheckpointMismatchError

from ...dto.embedding import KGEmbeddingModel, SubgraphRepresentation
from ...config import KGWrapperConfig, RevealGraph

from ..kg.graph_generator import GraphGenerator, EnvRep
from ..kg.embeddings import embed_graph, load_model
from ..kg.retrieval_methods import get_k_hop_subgraph, get_knn, get_random_walk_path


@dataclass
class KGObservation:
    """
    Class for the new observations of env wrapped with KGWrapper
    """
    state: Any
    subgraph: Optional[Graph] = None
    subgraph_embedding: Optional[Tuple[np.ndarray, np.ndarray]] = None
    knn_embedding: Optional[np.ndarray] = None  # array of shape knn_n, embedding_dim
    random_walk: Optional[np.array] = None  # array of indeces of entities and relations of shape (rw_k, 2)
    random_walk_embedding: Optional[np.ndarray] = None

    def to_dict(self):
        """Return a dict of Fields that are not `None`"""
        return {key: value for key, value in asdict(self).items() if value is not None}


class GraphHandler:
    """
    The `GraphHandler` provides all derivations from the graph (e.g.
    embedding...)
    """
    def __init__(
            self,
            config: KGWrapperConfig,
    ):
        self.graph: Graph = Graph()
        self.config = config

        self.model = None
        self._knn_embeddings = None

    @property
    def embedding_model(self) -> ERModel:
        return self.model.pipeline_results.model

    @property
    def entity_embeddings(self) -> pykeen.nn.Embedding:
        return self.embedding_model.entity_representations[0]

    @property
    def relation_embeddings(self) -> pykeen.nn.Embedding:
        return self.embedding_model.relation_representations[0]
    @property
    def embedding_tf(self) -> TriplesFactory:
        return self.model.triples_factory

    @property
    def embedding_results(self) -> KGEmbeddingModel:
        return KGEmbeddingModel(self.embedding_tf, self.model.pipeline_results)

    @property
    def knn_embeddings(self) -> SubgraphRepresentation:
        return self._knn_embeddings

    def reset(self, graph: Graph):
        """Reset the Graph Handler to a new Graph"""
        self.graph = graph
        self.model = self.get_embedding()
        if self.config.reveal == RevealGraph.KNN_EMBEDDING:
            self._knn_embeddings = get_knn(
                k=self.config.k_nn_k,
                model=self.model,
            )

    def get_embedding(self) -> KGEmbeddingModel:
        """
        Train a Knowledge Graph embedding on `self.kg`
        """
        renamed_old_checkpoint = False

        # if self.config.load_from_checkpoint and self.config.checkpoint.path is not None:
        #     print("load model from checkpoint")
        #     model = load_model(
        #         save_dir=self.config.checkpoint.path,
        #         checkpoint_name=self.config.checkpoint.name
        #     )
        #     return model

        def train():
            return embed_graph(
                graph=self.graph,
                model=self.config.model.architecture.value,
                epochs=self.config.train.n_epochs,
                split=self.config.train.split.as_tuple,
                save_dir=self.config.checkpoint.path,
                model_kwargs=self.config.model.as_dict,
                training_kwargs=self.config.train.as_dict,
                evaluation_kwargs=self.config.eval.as_dict,
                compute_embedding=self.config.compute_embeddings,  #do not compute embeddings if there is no need then we might just need the tf
            )

        try:
            model = train()
        except CheckpointMismatchError:
            os.remove(self.config.train.checkpoint_path)
            model = train()

        # we also need to save the triples_factory to actually use the saved model
        return model

    def get_random_walk(self, entity, ) -> np.ndarray:
        """Returns the normalized path of a random walk stariting from entity."""
        path = get_random_walk_path(self.graph, entity, self.config.rw_k)
        entities = np.array([self.embedding_tf.entity_to_id[str(ent)] if ent != -1 else -1 for ent in path.entities])/(self.embedding_tf.num_entities)
        relations = np.array([self.embedding_tf.relation_to_id[str(relation)] if relation != -1 else -1 for relation in path.relations])/(self.embedding_tf.num_relations)
        return np.transpose(np.column_stack((entities, relations)))  # the rescaled indices of the rw path

    def get_random_walk_embedding(self, entity, padding: Optional[Union[torch.Tensor, str]] = None) -> np.ndarray:
        """
        Returns the embeddings of entities and relations encountered during a RW.
        """
        path = get_random_walk_path(self.graph, entity, self.config.rw_k)
        if padding is None:
            pad_entity = self.embedding_tf.entity_to_id[str(entity)]
            pad_relation = 0
        # get ids
        entity_ids = torch.LongTensor([self.embedding_tf.entity_to_id[str(ent)] if ent != -1 else pad_entity for ent in path.entities])
        relation_ids = torch.LongTensor([self.embedding_tf.relation_to_id[str(relation)] if relation != -1 else pad_relation for relation in path.relations])

        entities = self.entity_embeddings(indices=entity_ids).detach().cpu().numpy()
        relations = self.relation_embeddings(indices=relation_ids).detach().cpu().numpy()
        return np.row_stack((entities, relations))


Observation = TypeVar("Observation", covariant=True)
KG_Observation = TypeVar("KG_Observation", bound=KGObservation)


class KGWrapper(gym.ObservationWrapper, Generic[Observation, EnvRep, KG_Observation]):
    """Base class for KGWrapper"""
    def __init__(
            self,
            env,
            config: Optional[KGWrapperConfig] = None,
            graph_generator: Optional[GraphGenerator] = None,
            graph_handler: Optional[GraphHandler] = None,
    ):
        super().__init__(env)
        if config is None:
            config = KGWrapperConfig()
        self.config = config

        assert graph_generator is not None, "A `GraphGenerator needs to be specified.`"
        self.graph_generator = graph_generator

        if graph_handler is not None:
            self.graph_handler = graph_handler
        else:
            graph_handler = GraphHandler
        self.graph_handler = graph_handler(
            config=self.config
        )

        # load the kg
        if config.kg_path is None:
            self.graph_handler.reset(self.get_environment_kg())
        else:
            self.graph_handler.reset(self.load_kg(self.config.kg_path))

        # adjust the observation space
        self.observation_space = self.get_observation_space()

    @abstractmethod
    def observation_to_kg_entity(self, observation: Observation) -> URIRef:
        """
        get an entity corresponding to the observation.
        todo: rdf2vec vs state2rdf
        """
        pass

    @abstractmethod
    def observation_to_kg_observation(self, observation: Observation) -> KG_Observation:
        pass

    @abstractmethod
    def get_env_representation(self) -> EnvRep:
        pass

    def get_environment_kg(self) -> Graph:
        """Get rdf representation of the maze and return rdf-lib graph"""
        # for a static environment we only need to generate the graph once
        env_rep = self.get_env_representation()
        self.graph_generator.generate_graph(env_rep)
        return self.graph_generator.domain_kg

    def load_kg(self, kg_path: str):
        try:
            g = Graph(kg_path)
            return g
        except Exception as ex:
            print(ex)
            print("Could not Load KG from the given path")
            return self.get_environment_kg()

    def get_observation_space(self) -> gym.spaces.Space:
        """
        Return the augmented Observation Space
        todo: observation space for returning the subgraph
        todo: adjust space for returning subgraph embeddings
        """
        default_spaces = {
            'state': self.env.observation_space,
            'subgraph': None,
            'random_walk': gym.spaces.box.Box(low=0, high=1, shape=(2, self.config.rw_k))
        }
        if self.config.compute_embeddings:
            default_spaces['subgraph_embedding'] = None,
            # k_nn_embeddings lie in a box with range equaling the embedding range
            default_spaces['knn_embedding'] = gym.spaces.box.Box(
                low=np.stack(
                    [
                        np.amin(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)
                        for _ in range(self.config.k_nn_k)
                    ]
                ),

                high=np.stack(
                    [
                        np.amax(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)
                        for _ in range(self.config.k_nn_k)
                    ],
                )
            )
            default_spaces['random_walk_embedding'] = gym.spaces.box.Box(  # todo: find a better representation of this state space maybe rescale to a -1,1 box
                low=np.stack(
                    [
                        np.amin(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)
                        for _ in range(self.config.rw_k)
                    ] + [
                        np.amin(
                            self.graph_handler.embedding_model.relation_representations[0]().detach().cpu().numpy() * 1.01, axis=0)
                        for _ in range(self.config.rw_k)
                    ]
                ),

                high=np.stack(
                    [
                        np.amax(self.graph_handler.embedding_model.entity_representations[0]().detach().cpu().numpy() * 1.01, axis=0)
                        for _ in range(self.config.rw_k)
                    ] + [
                        np.amax(
                            self.graph_handler.embedding_model.relation_representations[0]().detach().cpu().numpy() * 1.01, axis=0)
                        for _ in range(self.config.rw_k)
                    ]
                ),
                shape=(2 * self.config.rw_k, self.config.model.dim)  # two times rw_k as we have entities and relations
            )

        keys = self.config.get_observation_keys()
        return gym.spaces.Dict({k: v for k, v in default_spaces.items() if k in keys})

    def reveal_sub_kg(self, observation: Observation, depth: Optional[int] = None) -> Graph:
        """
        return all depth - hop triples of the environment KG as a subgraph.
        """
        if depth is None:
            depth = self.config.subgraph_depth
        node = self.observation_to_kg_entity(observation)
        return get_k_hop_subgraph(
            graph=self.graph_handler.graph,
            node=node,
            k=depth,
            directional=self.config.subgraph_directional,
        )

    def reveal_knn_embeddings(self, observation: Observation) -> np.ndarray:
        """
        Reveal the `self.config.reveal_k_nn_k` nearest neighbors of the state
        in the `self.kg_embedding_model`.
        """
        # get the URIRef of the state and the corresponding index w.r.t. the embedding model
        kg_entity = self.observation_to_kg_entity(observation)  # coordinates -> uri
        entity_id = self.graph_handler.embedding_tf.entities_to_ids([str(kg_entity)])  # uri -> node-id
        return self.graph_handler.knn_embeddings.entity_embeddings[entity_id].squeeze()

    def reveal_rw_path(self, observation) -> np.ndarray:
        return self.graph_handler.get_random_walk(entity=self.observation_to_kg_entity(observation))

    def reveal_rw_embedding(self, observation) -> np.ndarray:
        return self.graph_handler.get_random_walk_embedding(entity=self.observation_to_kg_entity(observation))

    def observation(self, observation: Observation, as_dict: bool = True) -> Union[KG_Observation, Dict[str, np.ndarray]]:
        """
                Augment the original observation in the env with a KG that can be
                observed in the current state

                :param observation: the observation returned by the plain env
                :param as_dict: set to false to return observation as KG_Observation
                :return: Knowledge graph Augmented Observation
                """
        observations = self.observation_to_kg_observation(observation)  # wraps current coordinates into an object
        if self.config.reveal == RevealGraph.SUBGRAPH:
            observations.observation_graph = self.reveal_sub_kg(observation)

        # todo: ad reveal_kg_embedding to reveal the embedding of the k-hop
        #  subgraph or return the subgraph as triples of indices -> shape
        #  (3, n_subgraph) Adjust the observation spaces then

        if self.config.reveal == RevealGraph.KNN_EMBEDDING:
            observations.knn_embedding = self.reveal_knn_embeddings(observation)
        elif self.config.reveal == RevealGraph.RANDOM_WALK:
            observations.random_walk = self.reveal_rw_path(observation)
        elif self.config.reveal == RevealGraph.RANDOM_WALK_EMBEDDING:
            observations.random_walk_embedding = self.reveal_rw_embedding(observation)

        if self.config.observation_as_dict and as_dict:
            return observations.to_dict()
        return observations
