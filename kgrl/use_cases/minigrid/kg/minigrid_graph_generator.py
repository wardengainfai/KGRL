import os
from typing import Optional, Tuple
from enum import Enum

import numpy as np
from rdflib import URIRef, Literal, Namespace

from ....__init__ import config
from ....base.kg.graph_generator import GraphGenerator

from ..utils import IDX_TO_STATE, IDX_TO_OBJECT, IDX_TO_COLOR, COLOR_TO_IDX, COMPASS

CELL_CLASS = URIRef("http://webprotege.stanford.edu/R7iydtyVZ70q2mr855dChbB")


class CellType(Enum):
    WALL = 'Wall'
    KEY = 'Key'
    EMPTY = 'Empty'
    DOOR = 'Door'
    GOAL = 'Goal'
    LAVA = 'Lava'
    BALL = 'Ball'
    BOX = 'Box'
    UNSEEN = 'Unseen'

    def uri(self, graph):
        if self == CellType.WALL:
            return graph.wall
        elif self == CellType.KEY:
            return graph.key
        elif self == CellType.EMPTY:
            return graph.empty
        elif self == CellType.DOOR:
            return graph.door
        elif self == CellType.BALL:
            return graph.ball
        elif self == CellType.BOX:
            return graph.box
        elif self == CellType.GOAL:
            return graph.goal
        elif self == CellType.LAVA:
            return graph.lava
        elif self == CellType.UNSEEN:
            return graph.unseen
        else:
            raise ValueError(f'Uri for cell type "{self.value}" is unknown')


class Direction(Enum):
    NORTH = 'North'
    EAST = 'East'
    SOUTH = 'South'
    WEST = 'West'

    def uri(self, graph):
        if self == Direction.NORTH:
            return graph.north
        elif self == Direction.EAST:
            return graph.east
        elif self == Direction.SOUTH:
            return graph.south
        elif self == Direction.WEST:
            return graph.west
        else:
            raise ValueError('Uri for direction "{self.value}" is unknown')


# the environment in Minigrid is represented as a env.grid.
class MinigridGraphGenerator(GraphGenerator[np.ndarray]):
    OBSTRUCTIONS = {"Wall", "Box", "Ball", "Key"}

    def __init__(self):
        super().__init__()
        self.ontology.parse(config.minigrid_ontology, format='turtle')

    def _bind_namespaces(self) -> None:
        super()._bind_namespaces()

        self.minigrid = Namespace('http://webprotege.stanford.edu/')
        self.ontology.bind("minigrid", self.minigrid)


        # todo: mapping entities like this is a bad idea! change it to be more general
        # properties

        self.is_located_in = self.minigrid.RB4ANTMp0eHRRNgK0pJTVmU
        self.has_color = self.minigrid.Rs8DERzwLe0LalQhxSSCNm

        # classes

        self.key = self.minigrid.RCadbGH3MoNbSG7fEauPj7y
        self.cell = self.minigrid.R7iydtyVZ70q2mr855dChbB
        self.wall = self.minigrid.RCq8WtAvuDN6ssGJG8KLKH5
        self.color = self.minigrid.R1TCHRfP0ZAnCeKS9WrKtX
        self.empty = self.minigrid.RvbVLdgbP5J9XemRdP0nUk
        self.door = self.minigrid.R8N6ULtlrgByQEMfNYMEJaj
        self.goal = self.minigrid.RrWxYEZm5Wz95sUeCImpUq
        self.lava = self.minigrid.RUiGFrV2njcolDOFBazeUs
        self.unseen = self.minigrid.RJ63OPH7mWovyVxkm1MUSw
        self.ball = self.minigrid.Rs9mkZeGYTUbwglDxLgNzQ
        self.box = self.minigrid.RDESLJr7Sb04qzqQbvX7hQs

        # directions

        self.north = self.minigrid.RDo4L7lRiQgXTyKMFazUEbL
        self.east = self.minigrid.RBFZa5nzQ4DVqNZDa5zM4dO
        self.south = self.minigrid.R8i3q5dRzmONG7M4Ca3roBT
        self.west = self.minigrid.R9Dtx5xehiW0h3VfEPUyENE

        # actions

        self.opens = self.minigrid.RC5vnGFeVwmOvBStNEwa2Uh

    def create_connection(self, s, p, o) -> None:
        # todo: create nodes from s, p, o if they are not nodes already
        s_node = s
        p_node = p
        o_node = o
        self.domain_kg.add((s_node, p_node, o_node))
        pass

    def get_cell_label(self, position: Tuple[int, int]) -> str:
        return ''.join(["cell_", str(position[0]), "_", str(position[1])])

    def get_cell_instance(self, position: Tuple[int, int]) -> URIRef:
        return URIRef("".join([self.minigrid, self.get_cell_label(position)]))

    def get_color_instance(self, color_label: str) -> URIRef:
        return URIRef("".join([self.minigrid, color_label]))

    def get_content_instance(self, content_type: str, position: Tuple[int, int]):
        """
        For now the instance name is represented with the initial position of the
        content and the content type
        """
        return URIRef("".join([self.minigrid, content_type.value, "_", str(position[0]), str(position[1])]))

    def _add_cell_to_kg(self, position: Tuple[int, int], rep: np.ndarray) -> None:
        """
        with all connections
        todo: refactor
        """
        cell_type: str = CellType(IDX_TO_OBJECT[rep[0]])
        cell_color: str = IDX_TO_COLOR[rep[1]]
        cell_state: int = rep[2]
        cell_label: str = self.get_cell_label(position)
        cell_instance = self.get_cell_instance(position)
        # skip cell if empty
        # todo: unfortunately "empty" cells can also be cells within the environment.
        #  thus we can not skiop these without loosing connectivity in the maze.
        # todo: modify the minigrid such that cells within the maze are floor and
        #  cells outside the reachable zone are empty
        if cell_type == "Unseen":
            return

        # add label
        self.create_connection(cell_instance, self.rdfs.label, Literal(cell_label))
        # add cell
        self.create_connection(
            cell_instance,
            self.rdfs.type,
            self.cell
        )

        # contents of the cell
        # content_type = self.get_subject(o=Literal(cell_type), p=self.rdfs.label)
        content_instance = self.get_content_instance(cell_type, position)
        self.create_connection(
            s=content_instance,
            p=self.rdfs.type,
            o=cell_type.uri(self)
        )
        self.create_connection(
            s=content_instance,
            p=self.is_located_in,
            o=cell_instance,
        )
        # content: add state so far only doors have a state different from 0: {"open": 0, "closed": 1, "locked":2}
        if cell_type == "Door":
            self.create_connection(
                s=content_instance,
                p=self.get_subject(o=Literal("hasState"), p=self.rdfs.label),
                o=self.get_subject(o=Literal(IDX_TO_STATE[cell_state]), p=self.rdfs.label),
            )

        # content: add color
        color_instance = self.get_color_instance(color_label=cell_color)
        self.create_connection(
            s=color_instance,
            p=self.rdfs.type,
            o=self.color,
        )
        self.create_connection(
            s=content_instance,
            p=self.has_color,
            o=color_instance,
        )

    def _has_connection(self, cell1_representation: np.ndarray, cell2_representation: np.ndarray) -> bool:
        """
        Return if there is potential for a connection between the two cells with
        representations.
        """
        type1 = IDX_TO_OBJECT[cell1_representation[0]]
        type2 = IDX_TO_OBJECT[cell2_representation[0]]
        obstructed = (type1 == "Door" and IDX_TO_STATE[cell1_representation[2]] in {"Closed", "Locked"})\
            or (type2 == "Door" and IDX_TO_STATE[cell2_representation[2]] in {"Closed", "Locked"})\
            or type1 in self.OBSTRUCTIONS \
            or type2 in self.OBSTRUCTIONS
        return not obstructed

    def _add_connection(self, cell1_position: Tuple[int, int], cell2_position: Tuple[int, int], direction: str) -> None:
        cell1_instance = self.get_cell_instance(cell1_position)
        cell2_instance = self.get_cell_instance(cell2_position)
        connection = Direction(direction).uri(self)

        self.create_connection(
            s=cell1_instance,
            p=connection,
            o=cell2_instance,
        )

    def _add_entity_interactions(self) -> None:
        """
        Add Interactions e.g. depending on color.
        """
        for color in COLOR_TO_IDX.keys():
            color_instance = self.get_color_instance(color_label=color)
            colored_subjects = list(self.domain_kg.subjects(predicate=self.has_color, object=color_instance))
            # add key - door connection
            doors = [subject for subject in colored_subjects if (subject, self.rdfs.type, self.door) in self.domain_kg]
            keys = [subject for subject in colored_subjects if (subject, self.rdfs.type, self.key) in self.domain_kg]
            for door in doors:
                for key in keys:
                    self.create_connection(
                        s=key,
                        p=self.opens,
                        o=door
                    )

    def generate_graph(self, env_representation: Optional[np.ndarray] = None) -> None:
        """
        Populate the `self.domain_kg`.

        :param env_representation: (e.g. `env.grid.encode()` to get encoding of
          a grid)
        :type: `np.ndarray`
        :return: `None`
        """
        # iterate over all cells
        rep_shape = env_representation.shape
        for i in range(rep_shape[0]):
            for j in range(rep_shape[1]):
                cell_position = (i, j)
                cell_rep = env_representation[cell_position]
                self._add_cell_to_kg(cell_position, cell_rep)  # with type, label, color etc
                # add connection to neighbors
                # a cell A is connected to another cell B if an agent can move from A to B
                # this mens closed and locked doors as well as walls and other obstructions
                # need to be taken into account.
                for direction_label, direction in COMPASS.items():
                    # get graph instance of the neighbor and make connection
                    neighbor_position = (cell_position[0] + direction[0], cell_position[1] + direction[1])
                    # position in bounds?
                    if neighbor_position[0] in range(rep_shape[0]) and neighbor_position[1] in range(rep_shape[1]):
                        neighbor_rep = env_representation[neighbor_position]
                        if self._has_connection(cell_rep, neighbor_rep):
                            self._add_connection(cell_position, neighbor_position, direction_label)
        self._add_entity_interactions()
