from typing import Dict, Tuple, Set

from rdflib import Literal, Namespace

from ....__init__ import config
from ....base.kg.graph_generator import GraphGenerator

from ..utils.MazeMap import MazeMap, MazeRoom, MazeWall

MazeRepresentation = Dict[Tuple[int, int], Dict[str, int]]


class MazeGraphGenerator(GraphGenerator):

    def __init__(self, path_to_ontology=config.maze_ontology):
        super().__init__()

        self.ontology.parse(path_to_ontology, format='turtle')

        self.north = self.get_labelled_subject('North')
        self.east = self.get_labelled_subject('East')
        self.south = self.get_labelled_subject('South')
        self.west = self.get_labelled_subject('West')

        self.turn_north = self.get_labelled_subject('TurnNorth')
        self.turn_east = self.get_labelled_subject('TurnEast')
        self.turn_south = self.get_labelled_subject('TurnSouth')
        self.turn_west = self.get_labelled_subject('TurnWest')

    def _bind_namespaces(self):
        super()._bind_namespaces()

        self.maze = Namespace('http://webprotege.stanford.edu/')
        self.ontology.bind("maze", self.maze)

        self.room = self.maze.RBdaVspmcugOCRE77I44T7v
        self.has_action = self.maze.R8EFdCdGqiXFOTbCaYXU3hD
        self.wall = self.maze.RijveJ9C8xu0oBM1iPC2zL

    def create_room(self, cell: MazeRoom):
        self.domain_kg.add((cell.uri, self.rdf.type, self.room))
        self.domain_kg.add((cell.uri, self.rdfs.label, Literal(cell.label)))

    def create_wall(self, cell: MazeWall):
        self.domain_kg.add((cell.uri, self.rdf.type, self.wall))
        self.domain_kg.add((cell.uri, self.rdfs.label, Literal(cell.label)))

    def try_create_wall(self, cell: MazeWall, seen_walls: Set[MazeWall]):
        if cell not in seen_walls:
            self.create_wall(cell)
            seen_walls.add(cell)

    def generate_graph(self, env_representation: MazeMap, include_walls: bool = False):
        seen_walls = set()

        for cell in env_representation.rooms:
            self.create_room(cell)

            if cell.no_north_wall:
                self.domain_kg.add((cell.uri, self.north, env_representation[(cell.x - 1, cell.y)].uri))
                self.domain_kg.add((cell.uri, self.has_action, self.turn_north))
            elif include_walls:
                self.domain_kg.add((cell.uri, self.north, (wall := cell.north_wall).uri))
                self.try_create_wall(wall, seen_walls)

            if cell.no_east_wall:
                self.domain_kg.add((cell.uri, self.east, env_representation[(cell.x, cell.y + 1)].uri))
                self.domain_kg.add((cell.uri, self.has_action, self.turn_east))
            elif include_walls:
                self.domain_kg.add((cell.uri, self.east, (wall := cell.east_wall).uri))
                self.try_create_wall(wall, seen_walls)

            if cell.no_south_wall:
                self.domain_kg.add((cell.uri, self.south, env_representation[(cell.x + 1, cell.y)].uri))
                self.domain_kg.add((cell.uri, self.has_action, self.turn_south))
            elif include_walls:
                self.domain_kg.add((cell.uri, self.south, (wall := cell.south_wall).uri))
                self.try_create_wall(wall, seen_walls)

            if cell.no_west_wall:
                self.domain_kg.add((cell.uri, self.west, env_representation[(cell.x, cell.y - 1)].uri))
                self.domain_kg.add((cell.uri, self.has_action, self.turn_west))
            elif include_walls:
                self.domain_kg.add((cell.uri, self.west, (wall := cell.west_wall).uri))
                self.try_create_wall(wall, seen_walls)
