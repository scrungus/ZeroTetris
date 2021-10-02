import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

Coords = List[List[Tuple[int, int]]]
Piece_info = Dict[str, Union[Coords, str]]


@dataclass
class PieceCoords:
    coords: Dict[int, Piece_info]

    def _get_piece_at_random(self) -> Tuple[Coords, int]:
        """
        Gets the coords of a piece selected uniformly at random.

        :return: the piece coords and id.
        """
        random_id = random.randint(0, len(self.coords.keys()) - 1)
        return self.coords[random_id]["coords"], random_id

    def _select_piece(self, idx: int) -> Coords:
        """
        Selects a piece using the ID provided.

        :param idx: the ID of the piece to be selected.
        """
        assert idx in list(self.coords.keys()), "Incorrect ID provided."
        return self.coords[idx]["coords"]
