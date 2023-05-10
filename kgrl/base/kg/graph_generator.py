import pprint
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from rdflib import Graph, Literal, Namespace


def print_ttl(graph):
    """Print the graph. If file_dir is provided the triples a re """
    for stmt in graph:
        pprint.pprint(stmt)


EnvRep = TypeVar("EnvRep")  # Type for representing the environment


class GraphGenerator(ABC, Generic[EnvRep]):
    def __init__(self):
        self.ontology: Graph = Graph()
        self.domain_kg: Graph = Graph()

        self._bind_namespaces()

    def _bind_namespaces(self) -> None:
        """Bind namespaces to the ontology."""
        # common namespaces
        self.owl = Namespace('http://www.w3.org/2002/07/owl#')
        self.rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        self.rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
        self.xml = Namespace('http://www.w3.org/XML/1998/namespace')
        self.xsd = Namespace('http://www.w3.org/2001/XMLSchema#')

        # Bind a few prefix, namespace pairs for pretty output
        self.ontology.bind("owl", self.owl)
        self.ontology.bind("rdf", self.rdf)
        self.ontology.bind("rdfs", self.rdfs)
        self.ontology.bind("xml", self.xml)
        self.ontology.bind("xsd", self.xsd)

        self.label = self.rdfs.label

    @abstractmethod
    def generate_graph(self, env_representation: EnvRep) -> None:
        """
        Populate the `self.domain_kg` Graph from the task-instance `env_representation`.
        """
        pass

    # @abstractmethod
    # def create_connection(self, s: Any, p: Any, o: Any) -> None:
    #     """Create a connection between `s` and `o` with relation `p`"""
    #     pass

    # def get_subject(self, o: str, p: str, g: Optional[Graph] = None) -> URIRef:
    #     """get subject value """
    #     graph = g if g is not None else self.ontology
    #     pred = URIRef(p)
    #     obj = Literal(o)
    #     s = graph.value(predicate=pred, object=obj)
    #     return s

    def get_labelled_subject(self, label: str):
        return self.ontology.value(predicate=self.label, object=Literal(label))

    def reset_domain_kg(self):
        self.domain_kg = Graph()

    def print_domain_kg(self):
        print_ttl(graph=self.domain_kg)
