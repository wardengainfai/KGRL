"""
Retrieve a random k-length random walk
"""
import random
from rdflib import Graph, URIRef
from ....dto.embedding import RWPath

def get_random_walk_path(graph: Graph, entity: URIRef, k: int) -> RWPath:
    predicates = []
    objects = []
    pad = False
    for i in range(k):
        matches = []
        if not pad:
            matches = list(graph.predicate_objects(subject=entity))
        if len(matches) >= 1:
            p, o = random.choice(matches)
            entity = o
        else:
            p, o = (-1, -1)
            pad = True
        predicates.append(p)
        objects.append(o)

    return RWPath(relations=predicates, entities=objects)
