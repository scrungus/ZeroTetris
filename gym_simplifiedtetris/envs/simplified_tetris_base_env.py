from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import gym
from gym.utils import seeding


class SimplifiedTetrisBaseEnv(gym.Env):
    """
    A class representing a simplified Tetris base environment, which ensures that all
    custom envs inherit from gym.Env and implement the essential methods and spaces.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (0, 4)

    def __init__(
        self, grid_dims: Sequence[int], piece_size: int, seed: Optional[int] = 8191
    ):
        if not isinstance(grid_dims, (list, tuple)) or len(grid_dims) != 2:
            raise TypeError(
                "Inappropriate format provided for grid_dims. It should be [int, int] or (int, int)."
            )

        assert piece_size in [
            1,
            2,
            3,
            4,
        ], "piece_size should be either 1, 2, 3, or 4."

        assert grid_dims[0] in list(
            range(piece_size + 1, 21)
        ), "Height must be an integer in the interval [piece_size + 1, 20]"
        assert grid_dims[1] in list(
            range(piece_size, 11)
        ), "Width must be an integer in the interval [piece_size, 10]."

        self._height_, self._width_ = grid_dims
        self._piece_size_ = piece_size

        self._num_actions_, self._num_pieces_ = {
            1: (grid_dims[1], 1),
            2: (2 * grid_dims[1] - 1, 1),
            3: (4 * grid_dims[1] - 4, 2),
            4: (4 * grid_dims[1] - 6, 7),
        }[piece_size]

        self._seed(seed=seed)

    def reset(self) -> np.array:
        return self._reset_()

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        return self._step_(action)

    def render(self, mode: Optional[str] = "human") -> np.ndarray:
        return self._render_(mode)

    def close(self) -> None:
        return self._close_()

    def _seed(self, seed: Optional[int] = 8191) -> None:
        self.np_random, _ = seeding.np_random(seed)

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError()

    @abstractmethod
    def _reset_(self):
        raise NotImplementedError()

    @abstractmethod
    def _step_(self, action):
        raise NotImplementedError()

    @abstractmethod
    def _render_(self, mode):
        raise NotImplementedError()

    @abstractmethod
    def _close_(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_terminal_reward(self):
        raise NotImplementedError()
