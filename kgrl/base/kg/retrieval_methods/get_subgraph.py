"""
Base retrieval Methods for getting subgraph
"""
from typing import Tuple

import torch
import numpy as np
from rdflib import Graph

from ....dto.embedding import KGEmbeddingModel, SubgraphRepresentation


def get_subgraph_triples() -> Graph:
   pass


def get_subgraph_representations(
        model: KGEmbeddingModel,
        subgraph: Graph,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get embedding of a subgraph of the graph used to train `model` with `tf`.
    :param model:
    :param tf:
    :param subgraph:
    :return: A tuple `(entity_embeddings, relation_embeddings)` containing the
     embedding vectors of the entities and the embeddings vectros of the
     relations
    """
    entity_representation_modules = model.pipeline_results.model.entity_representations  # List['pykeen.nn.RepresentationModule']
    relation_representation_modules = model.pipeline_results.model.relation_representations  # List['pykeen.nn.RepresentationModule']
    if len(entity_representation_modules) > 1 or len(relation_representation_modules) > 1:
        raise NotImplementedError("Not implemented for Embedding Models with multiple representation")

    entity_embeddings = entity_representation_modules[0]  # pykeen.nn.Embedding
    relation_embeddings = relation_representation_modules[0]  # pykeen.nn.Embedding

    entities_in_subgraph = [str(node) for node in subgraph.all_nodes()]
    relations_in_subgraph = {str(predicate) for predicate in subgraph.predicates(None, None)}
    entity_indices = model.triples_factory.entities_to_ids(entities_in_subgraph)
    relation_indices = model.triples_factory.relations_to_ids(relations_in_subgraph)
    return SubgraphRepresentation(
        entity_embeddings=entity_embeddings(torch.tensor(entity_indices)).detach().cpu().numpy(),
        relation_embeddings=relation_embeddings(torch.tensor(relation_indices)).detach().cpu().numpy(),
    )
