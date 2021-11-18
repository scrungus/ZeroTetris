from ...register import register
from ..simplified_tetris_part_binary_env import SimplifiedTetrisPartBinaryEnv
from .simplified_tetris_binary_shaped_env import SimplifiedTetrisBinaryShapedEnv


class SimplifiedTetrisPartBinaryShapedEnv(
    SimplifiedTetrisBinaryShapedEnv, SimplifiedTetrisPartBinaryEnv
):
    """
    A class representing a SimplifiedTetris env where the reward function is a
    scaled heuristic score and the obs space is the grid's part binary representation
    plus the current piece's id.

    :param grid_dims: the grid's dimensions.
    :param piece_size: the size of the pieces in use.
    :param seed: the rng seed.
    """

    pass


register(
    idx="simplifiedtetris-partbinary-shaped-v0",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryShapedEnv",
)
