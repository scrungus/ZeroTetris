from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces

from .simplified_tetris_engine import SimplifiedTetrisEngine
from .simplified_tetris_base_env import SimplifiedTetrisBaseEnv


class SimplifiedTetrisStandardEnv(SimplifiedTetrisBaseEnv):
    """
    A class representing a simplified Tetris environment, which implements the standard methods and action space.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of every piece.
    :param seed: the rng seed.
    """

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self._num_actions_)

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    def __init__(
        self, grid_dims: Sequence[int], piece_size: int, seed: Optional[int] = 8191
    ):
        super().__init__(grid_dims, piece_size, seed)

        self._engine = SimplifiedTetrisEngine(
            grid_dims=grid_dims,
            piece_size=piece_size,
            num_pieces=self._num_pieces_,
            num_actions=self._num_actions_,
        )

    def __str__(self) -> str:
        return np.array(self._engine._grid.T, dtype=int).__str__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(({self._height_!r}, {self._width_!r}), {self._piece_size_!r})"

    def _reset_(self) -> np.array:
        self._engine._reset()
        return self._get_obs()

    def _step_(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        """
        Hard drops the current piece according to the argument provided. Terminates the game if a condition is met. Otherwise, a new piece is selected, and the anchor is reset.

        :param action: the action to be taken.
        :return: the next observation, reward, game termination indicator, and env info.
        """
        info = {}

        translation, self._engine._rotation = self._engine._all_available_actions[
            self._get_obs()[-1]
        ][action]

        self._engine._anchor = [translation, self._piece_size_ - 1]

        self._engine._hard_drop()
        self._engine._update_grid(True)

        # The game terminates when any of the dropped piece's blocks occupies any of the top 'piece_size' rows, before any full rows are cleared.
        if np.any(self._engine._grid[:, : self._piece_size_]):
            info["num_rows_cleared"] = 0
            self._engine._final_scores = np.append(
                self._engine._final_scores, self._engine._score
            )
            return self._get_obs(), self._get_terminal_reward(), True, info

        reward, num_rows_cleared = self._get_reward()
        self._engine._score += num_rows_cleared
        self._engine._update_coords_and_anchor()
        info["num_rows_cleared"] = num_rows_cleared

        return self._get_obs(), float(reward), False, info

    def _render_(self, mode: Optional[str] = "human") -> np.ndarray:
        return self._engine._render(mode)

    def _close_(self) -> None:
        return self._engine._close()

    def _get_reward(self) -> Tuple[float, int]:
        return self._engine._get_reward()

    def _get_terminal_reward(self) -> float:
        return 0.0

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError()
