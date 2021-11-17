from abc import abstractmethod
from typing import Optional, Sequence

import gym
from gym.utils import seeding


class SimplifiedTetrisBaseEnv(gym.Env):
    """
    A class representing a simplified Tetris base environment, which ensures that all
    custom envs inherit from gym.Env and implement the required methods and spaces.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (0, 4)

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def render(self, mode):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    def __init__(
        self, grid_dims: Sequence[int], piece_size: int, seed: Optional[int] = 8191
    ):
        assert piece_size in [
            1,
            2,
            3,
            4,
        ], "Size of piece should be either 1, 2, 3, or 4."
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

        # Seed the rng.
        self._seed(seed=seed)

    def _seed(self, seed: Optional[int] = 8191) -> None:
        self.np_random, _ = seeding.np_random(seed)
