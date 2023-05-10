from typing import List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
from rdflib import Graph, URIRef
from pydantic import BaseModel

from pykeen.pipeline import PipelineResult
from pykeen.triples import TriplesFactory


# todo: add validators to custom types to enable typechecking with pydantic (in conjunction with BaseModel inheritance)


@dataclass
class KGEmbeddingModel:
    triples_factory: TriplesFactory
    pipeline_results: Optional[PipelineResult] = None


@dataclass
class SubgraphRepresentation:
    entity_embeddings: np.ndarray
    entity_embedding_indices: Optional[List[int]] = field(default_factory=list)
    relation_embeddings: Optional[np.ndarray] = None  # todo: find a better default factory
    relation_embedding_indices: Optional[List[int]] = field(default_factory=list)


@dataclass
class Subgraph:
    subgraph: Graph


@dataclass()
class RWPath:
    relations: List[Union[URIRef, int]]
    entities: List[Union[URIRef, int]]

class GraphFragments(BaseModel):
    entities: List[str]
    entities_indices: List[int]
    relations: List[str]
    relations_indices: List[int]
