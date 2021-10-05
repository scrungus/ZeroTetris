import numpy as np

from gym import spaces

from gym_simplifiedtetris.envs.simplified_tetris_binary_env import (
    SimplifiedTetrisBinaryEnv,
)
from gym_simplifiedtetris.register import register


class SimplifiedTetrisPartBinaryEnv(SimplifiedTetrisBinaryEnv):
    """
    A class representing a Tetris environment, where observation space is the
    grid's binary representation excluding the top piece_size rows, plus the
    current piece's id.
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

    def _get_obs_(self) -> np.array:
        """
        Gets the current observation, which is the grid's binary representation excluding the
        top piece_size rows, plus the current piece's id.

        :return: the current observation.
        """
        current_grid = self._engine._grid[:, self._piece_size_ :].flatten()
        return np.append(current_grid, self._engine._current_piece_id)


register(
    idx="simplifiedtetris-partbinary-v0",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisPartBinaryEnv",
)
