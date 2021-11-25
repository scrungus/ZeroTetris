from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .all_pieces import PIECES_DICT

PieceCoord = List[Tuple[int, int]]
Rotation = int
PieceCoords = Dict[Rotation, PieceCoord]


def _generate_max_min(coord_string: str, coords: PieceCoords):
    coord_strings = {
        "max_y_coord": {"func": np.max, "index": 1},
        "min_y_coord": {"func": np.min, "index": 1},
        "max_x_coord": {"func": np.max, "index": 0},
        "min_x_coord": {"func": np.min, "index": 0},
    }
    return {
        rot: coord_strings[coord_string]["func"](
            [coord[coord_strings[coord_string]["index"]] for coord in coords]
        )
        for rot, coords in coords.items()
    }


@dataclass
class Piece(object):
    _size: int
    _idx: int
    _rotation: int = 0
    _all_coords: PieceCoords = field(init=False)
    _coords: PieceCoord = field(init=False)
    _name: str = field(init=False)
    _max_y_coord: Dict[int, int] = field(init=False)
    _min_y_coord: Dict[int, int] = field(init=False)
    _max_x_coord: Dict[int, int] = field(init=False)
    _min_x_coord: Dict[int, int] = field(init=False)

    def __post_init__(self):
        self._all_coords = deepcopy(PIECES_DICT[self._size][self._idx]["coords"])
        self._coords = self._all_coords[self._rotation]
        self._name = deepcopy(PIECES_DICT[self._size][self._idx]["name"])
        self._max_y_coord = _generate_max_min("max_y_coord", self._all_coords)
        self._min_y_coord = _generate_max_min("min_y_coord", self._all_coords)
        self._max_x_coord = _generate_max_min("max_x_coord", self._all_coords)
        self._min_x_coord = _generate_max_min("min_x_coord", self._all_coords)
