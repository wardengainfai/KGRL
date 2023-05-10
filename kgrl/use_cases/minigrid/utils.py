# Utility functions for the minigrid environment
from typing import Tuple, List, Set

import numpy as np

# modifeid IDX_TO_COLOR and IDX_TO_OBJECT to fit the labels in the kg
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    "Unseen": 0,
    "Empty": 1,
    "Wall": 2,
    "Room": 3,  # is floor in minigrid
    "Door": 4,
    "Key": 5,
    "Ball": 6,
    "Box": 7,
    "Goal": 8,
    "Lava": 9,
    "Agent": 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

STATE_TO_IDX = {
    "Open": 0,
    "Closed": 1,
    "Locked": 2,
}

IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

COMPASS = {
        "North": (0, -1),
        "East": (-1, 0),
        "South": (0, 1),
        "West": (1, 0)
    }  # the playing field is again transposed from the view...


def improve_encoding(encoding: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    The encoding in the minigrid environments lacks specifity when it comes to
    distinguishing cell inside the maze and those that are not reachable at all.

    This function takes the encoding of a Mnigrid environment and returns an
    encoding where cells outside the reach of the agent are allways set to `unseen`
    (with index 0).
    """
    # there could be a multiple goals
    goals: List[Tuple[int, int]] = [tuple(cell) for cell in np.argwhere(encoding[:, :, 0] == OBJECT_TO_IDX["Goal"])]

    def connected_cells(cell: Tuple[int, int]) -> Set[Tuple[int, int]]:
        ngbh = set()
        for direction in COMPASS.values():
            cell_position = (cell[0] + direction[0], cell[1] + direction[1])
            if encoding[cell_position][0] != OBJECT_TO_IDX["Wall"]:
                ngbh.add(cell_position)
        return ngbh
    inside = set()
    while goals:
        goal = goals.pop()
        unvisited = {goal}
        connected_component = set()
        while unvisited:
            current = unvisited.pop()
            unvisited.update(connected_cells(current))
            connected_component.add(current)
            unvisited.difference_update(connected_component)
        inside.update(connected_component)

    if not inplace:
        encoding = encoding.copy()

    for i in range(encoding.shape[0]):
        for j in range(encoding.shape[1]):
            if (i, j) not in inside and encoding[(i, j)][0] != OBJECT_TO_IDX["Wall"]:
                encoding[(i,j)] = 0
    return encoding
