from rdflib import Graph, URIRef

from ....dto.embedding import Subgraph


def get_k_hop_subgraph(
        graph: Graph,
        node: URIRef,
        k: int,
        directional: bool = True
) -> Subgraph:
    """
    Return the k-hop subgraph of graph starting from node.

    :param graph: Graph from which the subgraph is constructed.
    :param node: starting node.
    :param k: number of hops from the starting node.
    :param directional: If set to `True` only outbound connections from a node
      are considered
    :return: Graph.
    """
    visited = []
    current = {node}
    triples = set()
    for i in range(k):
        while current:
            s = current.pop()
            visited.append(s)
            triples.update(graph.triples((s, None, None)))
            if not directional:
                triples.update(graph.triples((None, None, s)))

        current = {o for s, p, o in triples}.union({s for s, p, o in triples})  # don't have to distinguish directional here since if directional than only visited entities get added in the union
        current.difference_update(visited)  # subtract the entities that have been visited
    sub_graph = Graph()
    for triple in triples:
        sub_graph.add(triple)
    return Subgraph(subgraph=sub_graph)
