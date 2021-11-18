import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

Coords = Union[List[Tuple[int, int]], Dict[int, List[Tuple[int, int]]]]
PieceInfo = Dict[str, Union[Coords, str]]


@dataclass
class Pieces(object):
    _info: Dict[int, PieceInfo]

    def _get_piece_info_at_random(self) -> Tuple[Coords, int]:
        """
        Gets the coords of a piece selected uniformly at random.

        :return: the piece coords and id.
        """
        random_id = random.randint(0, len(self._info.keys()) - 1)
        return self._info[random_id], random_id

    def _select_piece_info(self, idx: int) -> Coords:
        """
        Selects a piece using the ID provided.

        :param idx: the ID of the piece to be selected.
        """
        assert idx in list(self._info.keys()), "Incorrect ID provided."
        return self._info[idx]
