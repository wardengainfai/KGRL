from enum import Enum


class RevealGraph(Enum):
    SUBGRAPH = 'subgraph'
    SUBGRAPH_EMBEDDING = 'subgraph_embedding'
    KNN_EMBEDDING = 'knn_embedding'
    RANDOM_WALK = 'random_walk'
    RANDOM_WALK_EMBEDDING = 'random_walk_embedding'
