"""
This module contains methods around the pykeen triples factories
"""
import numpy as np
from rdflib import Graph

from pykeen.triples import TriplesFactory


def triples_factory_from_graph(graph: Graph) -> TriplesFactory:
    triples = np.array(list(graph.triples((None, None, None))))
    return TriplesFactory.from_labeled_triples(triples)
