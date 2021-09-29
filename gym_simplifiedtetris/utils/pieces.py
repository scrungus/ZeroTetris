import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PieceCoords:
    coords: dict

    def get_piece_at_random(self) -> Tuple[list, int]:
        """
        Gets the coords of a piece selected uniformly at random.

        :return: the piece coords and id.
        """
        random_id = random.randint(1, len(self.coords.keys()))
        return self.coords[random_id]['coords'], random_id

    def select_piece(self, idx: int):
        """
        Selects a piece using the ID provided.

        :param idx: the ID of the piece to be selected.
        """
        assert idx in list(self.coords.keys()), 'Incorrect ID provided.'
        return self.coords[idx]['coords']
