from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from itertools import compress
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from rdflib import Graph
from kgrl.dto.embedding import Subgraph


class Masker(metaclass=ABCMeta):
    """
    TODO: Doc
    """

    @abstractmethod
    def mask(self, graph: Graph) -> np.ndarray:
        """
        TODO: Doc
        """
        pass

    def __and__(self, other: "Masker") -> "Masker":
        return FunMasker(
            lambda graph: self.mask(graph) & other.mask(graph),
        )

    def __or__(self, other: "Masker") -> "Masker":
        return FunMasker(
            lambda graph: self.mask(graph) | other.mask(graph),
        )

    def __invert__(self) -> "Masker":
        return FunMasker(
            lambda graph: ~self.mask(graph),
        )


@dataclass
class FunMasker(Masker):
    """
    TODO: Doc
    """

    fun: Callable[[Graph], np.ndarray]

    def mask(self, graph: Graph) -> np.ndarray:
        """
        TODO: Doc
        """
        return self.fun(graph)


@dataclass
class BaseMasker(Masker):
    """
    TODO: Doc
    """

    fraction_to_keep: float

    def mask(self, graph: Graph) -> np.ndarray:
        """
        TODO: Doc
        """
        return np.random.rand(len(graph)) < self.fraction_to_keep


@dataclass
class TripletFunMasker(Masker):
    """
    TODO: Doc
    """

    fun: Callable[[Tuple[Any, Any, Any]], bool]

    def mask(self, graph: Graph) -> np.ndarray:
        """
        TODO: Doc
        """
        return np.array([self.fun(triplet) for triplet in graph])


def mask_graph(
    graph: Graph,
    p: float = 0.5,
    masker: Optional[Union[Masker, Callable[[Tuple[Any, Any, Any]], bool]]] = None,
) -> Subgraph:
    """
    TODO: Doc
    """
    if masker is None:
        masker = BaseMasker(p)
    elif isinstance(masker, Callable):
        masker = TripletFunMasker(masker)
    mask = masker.mask(graph)
    new_graph = Graph()
    for triple in compress(graph, mask):
        new_graph.add(triple)
    return Subgraph(subgraph=new_graph)
