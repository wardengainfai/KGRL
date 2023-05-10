from rdflib import Graph
from pydantic import BaseModel


class GraphGenerator(BaseModel):
    graph: Graph
