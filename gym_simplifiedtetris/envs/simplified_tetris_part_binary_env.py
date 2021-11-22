import numpy as np
from gym import spaces

from ..register import register
from .simplified_tetris_standard_env import SimplifiedTetrisStandardEnv


class SimplifiedTetrisPartBinaryEnv(SimplifiedTetrisStandardEnv):
    """
    A class representing a Tetris environment, where the observation space is a flattened NumPy array containing the grid's binary representation excluding the top piece_size rows, plus the current piece's id.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.append(
                np.zeros(self._width_ * (self._height_ - self._piece_size_)), 0
            ),
            high=np.append(
                np.ones(self._width_ * (self._height_ - self._piece_size_)),
                self._num_pieces_ - 1,
            ),
            dtype=np.int,
        )

    def _get_obs(self) -> np.array:
        """
        Gets the current observation, which is a flattened NumPy array containing the grid's binary representation excluding the top piece_size rows, plus the current piece's id.

        :return: the current observation.
        """
        current_grid = self._engine._grid[:, self._piece_size_ :].flatten()
        return np.append(current_grid, self._engine._current_piece_id)


register(
    idx="simplifiedtetris-partbinary-v0",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryEnv",
)
