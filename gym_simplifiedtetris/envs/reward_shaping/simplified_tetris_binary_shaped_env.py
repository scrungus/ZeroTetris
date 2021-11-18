from typing import Optional, Sequence

from ...register import register
from ..simplified_tetris_binary_env import SimplifiedTetrisBinaryEnv
from .simplified_tetris_shaping_reward import SimplifiedTetrisShapingReward


class SimplifiedTetrisBinaryShapedEnv(
    SimplifiedTetrisShapingReward, SimplifiedTetrisBinaryEnv
):
    """
    A class representing a Tetris environment, where the reward function is a
    potential-based shaping reward and the observation space is the grid's binary repr plus the current piece's id.

    :param grid_dims: the grid's dimensions.
    :param piece_size: the size of the pieces in use.
    :param seed: the rng seed.
    """

    def __init__(
        self, grid_dims: Sequence[int], piece_size: int, seed: Optional[int] = 8191
    ):
        super().__init__()
        SimplifiedTetrisBinaryEnv.__init__(self, grid_dims, piece_size, seed)


register(
    idx="simplifiedtetris-binary-shaped-v0",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisBinaryShapedEnv",
)
