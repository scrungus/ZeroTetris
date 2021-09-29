from abc import abstractmethod
from typing import Tuple

import numpy as np
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
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            grid_dims: Tuple[int, int],
            piece_size: int,
            seed: int = 8191,
    ):
        assert piece_size in [
            1, 2, 3, 4], 'Size of piece should be either 1, 2, 3, or 4.'
        assert grid_dims[0] in list(range(
            piece_size + 1, 21)), 'Height must be an integer in the interval [piece_size + 1, 20]'
        assert grid_dims[1] in list(range(
            piece_size, 11)), 'Width must be an integer in the interval [piece_size, 10].'

        self.height, self.width = grid_dims
        self.piece_size = piece_size

        self.num_actions, self.num_pieces = {
            1: (grid_dims[1], 1),
            2: (2 * grid_dims[1] - 1, 1),
            3: (4 * grid_dims[1] - 4, 2),
            4: (4 * grid_dims[1] - 6, 7)
        }[piece_size]
        self.REWARD_RANGE = (0, 4)

        # Seed the rng.
        self.seed(seed=seed)

    def reset(self) -> np.array:
        return self._reset_()

    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        return self._step_(action)

    def render(self, mode: str = 'human') -> np.ndarray:
        return self._render_(mode)

    def close(self):
        return self._close_()

    def seed(self, seed: int = 8191):
        self.np_random, _ = seeding.np_random(seed)

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def _close_(self):
        raise NotImplementedError()

    @abstractmethod
    def _render_(self, mode: str):
        raise NotImplementedError()

    @abstractmethod
    def _reset_(self):
        raise NotImplementedError()

    @abstractmethod
    def _step_(self, action: int):
        raise NotImplementedError()
