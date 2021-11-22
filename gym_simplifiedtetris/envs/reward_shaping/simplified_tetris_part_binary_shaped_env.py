from typing import Optional, Sequence

from ...register import register
from ..simplified_tetris_part_binary_env import SimplifiedTetrisPartBinaryEnv
from .simplified_tetris_shaping_reward import SimplifiedTetrisShapingReward


class SimplifiedTetrisPartBinaryShapedEnv(
    SimplifiedTetrisShapingReward, SimplifiedTetrisPartBinaryEnv
):
    """
    A class representing a SimplifiedTetris env where the reward function is a
    scaled heuristic score and the obs space is the grid's part binary representation plus the current piece's id.

    :param grid_dims: the grid's dimensions.
    :param piece_size: the size of the pieces in use.
    :param seed: the rng seed.
    """

    def __init__(
        self, grid_dims: Sequence[int], piece_size: int, seed: Optional[int] = 8191
    ):
        super().__init__()
        SimplifiedTetrisPartBinaryEnv.__init__(self, grid_dims, piece_size, seed)


register(
    idx="simplifiedtetris-partbinary-shaped-v0",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryShapedEnv",
)
