import numpy as np
from gym import spaces

from ..register import register
from .simplified_tetris_standard_env import SimplifiedTetrisStandardEnv


class SimplifiedTetrisBinaryEnv(SimplifiedTetrisStandardEnv):
    """
    A class representing a custom Gym env for Tetris, where the observation space
    is the binary representation of the grid plus the current piece's id.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.append(np.zeros(self._width_ * self._height_), 0),
            high=np.append(
                np.ones(self._width_ * self._height_), self._num_pieces_ - 1
            ),
            dtype=np.int,
        )

    def _get_obs(self) -> np.array:
        current_grid = self._engine._grid.flatten()
        return np.append(current_grid, self._engine._current_piece_id)


register(
    idx="simplifiedtetris-binary-v0",
    entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv",
)
