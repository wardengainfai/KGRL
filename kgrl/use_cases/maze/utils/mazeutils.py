from typing import Dict, Optional, List, Tuple

import numpy as np
from pyamaze import maze
import gym


def pyamaze2gymmaze(pyamaze_map: Dict[Tuple[int, int], Dict[str, int]],) -> np.ndarray:
    """
    Transform a pyamaze style map to a gym_maze style map.
    :param pyamaze_map:
    :return: gym_maze style map.
    """
    walls_dict = {"N": 0x1, "E": 0x2, "S": 0x4, "W": 0x8}
    l = list(pyamaze_map.keys())
    map_range = [
        [max(l, key=lambda x: x[0])[0], min(l, key=lambda x: x[0])[0]],
        [max(l, key=lambda x: x[1])[1], min(l, key=lambda x: x[1])[1]],
    ]
    map_shape = [
        map_range[0][0]-map_range[0][1]+1,
        map_range[1][0]-map_range[1][1]+1,
    ]
    gymmaze = np.zeros(map_shape, dtype=np.int64)
    for key, value in pyamaze_map.items():
        # in pyamaze directions with walls have 0
        # in gymmaze directions wih walls have 0
        gymmaze[(key[0]-map_range[0][1], key[1]-map_range[1][1])] = sum([walls_dict[k] for k, v in value.items() if v == 1])
    return gymmaze.transpose()  # maze_cells in gymmaze are transposed.


def gymmaze2pyamaze(
        gymmaze: np.ndarray,
        adjust: bool = False,
        check_boundary: bool = True,
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    Transform a gymmaze style map (`np.ndarray`) to a pyamaze stye map.
    :param gymmaze:
    :param adjust: If `True` adjust notation to pyamaze (maps start in `(1,1)`)
    :param check_boundary: If True maze will have walls on the boundary.
    :return: pyamaze syle map.
    """
    gymmaze = gymmaze.transpose()  # the maze cells in gym maze are transposed !!
    walls = lambda cell: {
        "N": ((cell & 0x1) >> 0),  # flipping the values of walls to adjust for the different map notation
        "E": ((cell & 0x2) >> 1),
        "S": ((cell & 0x4) >> 2),
        "W": ((cell & 0x8) >> 3),
    }
    pyamaze_map = {}
    for i in range(gymmaze.shape[0]):
        for j in range(gymmaze.shape[1]):
            key = (i+1, j+1) if adjust else (i, j)
            pyamaze_map[key] = walls(gymmaze[(i, j)])
            if check_boundary:
                if i == 0:
                    pyamaze_map[key]["N"] = 0
                elif i == gymmaze.shape[0]-1:
                    pyamaze_map[key]["S"] = 0
                if j == 0:
                    pyamaze_map[key]["W"] = 0
                elif j == gymmaze.shape[1]-1:
                    pyamaze_map[key]["E"] = 0

    return pyamaze_map


def show_gym_registry(string: Optional[str] = None, return_list: bool = False) -> Optional[List[str]]:
    spec_id_list = [spec.id for spec in gym.envs.registry.all() if (string in spec.id.lower() if string is not None else True)]
    print('\n'.join(spec_id_list))
    if return_list:
        return spec_id_list


if __name__ == "__main__":
    shape = [5, 5]
    m = maze(*shape)
    m.CreateMaze()
    pyamaze_map = m.maze_map

    # convert to a gymmaze maze - a np.ndarray of shape
    gymmaze_map = pyamaze2gymmaze(pyamaze_map)
