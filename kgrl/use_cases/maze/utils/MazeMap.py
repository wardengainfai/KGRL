from typing import Tuple

import numpy as np
from rdflib import URIRef

WALL = '#'


class MazeWall:
    def __init__(self, north_room_label: str = None, east_room_label: str = None, south_room_label: str = None, west_room_label: str = None):
        self.north_room = north_room_label
        self.east_room = east_room_label
        self.south_room = south_room_label
        self.west_room = west_room_label

    @property
    def label(self) -> str:
        if self.west_room is not None and self.east_room is not None:
            return f'wall_between_{self.west_room}_and_{self.east_room}'
        elif self.north_room is not None and self.south_room is not None:
            return f'wall_between_{self.north_room}_and_{self.south_room}'
        elif self.north_room is not None:
            return f'south_wall_{self.north_room}'
        elif self.south_room is not None:
            return f'north_wall_{self.south_room}'
        elif self.west_room is not None:
            return f'east_wall_{self.west_room}'
        elif self.east_room is not None:
            return f'west_wall_{self.east_room}'
        else:
            raise ValueError('Cannot generate label - there are not adjacent rooms')

    @property
    def uri(self):
        return URIRef(f'http://webprotege.stanford.edu/{self.label}')


class MazeRoom:
    def __init__(self, x: int, y: int, north_wall: bool = False, east_wall: bool = False, south_wall: bool = False, west_wall: bool = False):
        self.x = x
        self.y = y
        
        self.no_north_wall = north_wall
        self.no_south_wall = south_wall
        self.no_east_wall = east_wall
        self.no_west_wall = west_wall

        self.north_wall = None
        self.south_wall = None
        self.east_wall = None
        self.west_wall = None

    @property
    def label(self) -> str:
        return f'cell_{self.x}_{self.y}'

    @property
    def uri(self):
        return URIRef(f'http://webprotege.stanford.edu/{self.label}')

    def __str__(self):
        return "\n".join((
            f'{WALL if not self.north_wall or not self.west_wall else " "}{WALL if not self.north_wall else " "}{WALL if not self.north_wall or not self.east_wall else " "}',
            f'{WALL if not self.west_wall else " "} {WALL if not self.east_wall else " "}',
            f'{WALL if not self.south_wall or not self.west_wall else " "}{WALL if not self.south_wall else " "}{WALL if not self.south_wall or not self.east_wall else " "}'
        ))


def decode_walls(matrix, x: int, y: int):
    walls = matrix[x, y]
    decoded = (
        x > 0 and ((walls & 0x1) >> 0) > 0,
        y < matrix.shape[0] and ((walls & 0x2) >> 1) > 0,
        x < matrix.shape[0] and ((walls & 0x4) >> 2) > 0,
        y > 0 and ((walls & 0x8) >> 3) > 0
    )
    return decoded


def setup_walls(rooms: Tuple[MazeRoom], length: int, height: int):

    # Set up vertical walls

    for x in range(height):
        for y in range(-1, length):
            if y == -1:  # Leftmost cell with only one wall from the west side
                west_wall = MazeWall(east_room_label = (room := rooms[x * length]).label)
                room.west_wall = west_wall
                continue
            if y == length - 1:  # Leftmost cell with only one wall from the west side
                east_wall = MazeWall(west_room_label = (room := rooms[length - 1 + x * length]).label)
                room.east_wall = east_wall
                continue

            west_room = rooms[y + x * length]
            east_room = rooms[y + 1 + x * length]

            if not west_room.no_east_wall:
                assert not east_room.no_west_wall

                wall = MazeWall(west_room_label = west_room.label, east_room_label = east_room.label)

                west_room.east_wall = wall
                east_room.west_wall = wall

    # Set up horizontal walls

    for x in range(-1, height):
        for y in range(length):
            if x == -1:  # Leftmost cell with only one wall from the west side
                north_wall = MazeWall(south_room_label = (room := rooms[y]).label)
                room.north_wall = north_wall
                continue
            if x == height - 1:  # Leftmost cell with only one wall from the west side
                south_wall = MazeWall(north_room_label = (room := rooms[y + (height - 1) * length]).label)
                room.south_wall = south_wall
                continue

            north_room = rooms[y + x * length]
            south_room = rooms[y + (x + 1) * length]

            if not north_room.no_south_wall:
                assert not south_room.no_north_wall

                wall = MazeWall(south_room_label = south_room.label, north_room_label = north_room.label)

                north_room.south_wall = wall
                south_room.north_wall = wall


class MazeMap:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self._rooms = None

    @property
    def rooms(self):
        if self._rooms is None:
            matrix = self.matrix.transpose()  # the maze cells in gym maze are transposed !!

            rooms = [
                MazeRoom(x, y, *decode_walls(matrix, x, y))
                for x in range(matrix.shape[0])
                for y in range(matrix.shape[1])
            ]

            setup_walls(rooms, matrix.shape[0], matrix.shape[1])

            self._rooms = rooms

        return self._rooms

    def __getitem__(self, coordinates):
        x, y = coordinates
        return self.rooms[self.matrix.transpose().shape[1] * x + y]

    @staticmethod
    def from_dict(cells: dict):
        cells_ = []
        size = sorted(cells.keys())[-1]

        for cell in cells.values():
            cells_.append(
                cell['N'] + cell['E'] * 2 + cell['S'] * 4 + cell['W'] * 8
            )

        return MazeMap(np.array(cells_).reshape(size))
