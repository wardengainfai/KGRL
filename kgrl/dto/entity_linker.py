import numpy as np
from rdflib import Graph, URIRef
from pykeen.pipeline import PipelineResult
from dataclasses import dataclass
from typing import List

class EntityRelationLinker:
    kg: Graph
    text: str

class LinkerEmbeddings(EntityRelationLinker):
    n_triples: int
    kg_emb: PipelineResult

@dataclass
class TripleSimilarity:
    triple: List[URIRef]
    similarity: float

@dataclass
class TripleEmbeddings:
    triple_embeddings: np.ndarray
    confidence: np.ndarray
