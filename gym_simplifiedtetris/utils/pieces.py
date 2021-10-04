import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

coords = List[List[Tuple[int, int]]]
piece_info = Dict[str, Union[coords, str]]


@dataclass
class PiecesInfo:
    info: Dict[int, piece_info]

    def _get_piece_at_random(self) -> Tuple[coords, int]:
        """
        Gets the coords of a piece selected uniformly at random.

        :return: the piece coords and id.
        """
        random_id = random.randint(0, len(self.info.keys()) - 1)
        return self.info[random_id]["coords"], random_id

    def _select_piece(self, idx: int) -> coords:
        """
        Selects a piece using the ID provided.

        :param idx: the ID of the piece to be selected.
        """
        assert idx in list(self.info.keys()), "Incorrect ID provided."
        return self.info[idx]["coords"]
