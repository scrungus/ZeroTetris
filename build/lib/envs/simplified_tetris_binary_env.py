from typing import Any, Dict, Sequence, Tuple

import numpy as np
from gym import spaces
from gym_simplifiedtetris.envs.simplified_tetris_base_env import SimplifiedTetrisBaseEnv
from gym_simplifiedtetris.envs.simplified_tetris_engine import SimplifiedTetrisEngine
from gym_simplifiedtetris.register import register


class SimplifiedTetrisBinaryEnv(SimplifiedTetrisBaseEnv):
    """
    A class representing a custom Gym env for Tetris, where the observation space
    is the binary representation of the grid plus the current piece's id.

    :param grid_dims: the grid dimensions.
    :param piece_size: the size of the pieces in use.
    """

    def __init__(
        self,
        grid_dims: Sequence[int],
        piece_size: int,
    ):
        super(SimplifiedTetrisBinaryEnv, self).__init__(
            grid_dims=grid_dims,
            piece_size=piece_size,
        )

        self._engine = SimplifiedTetrisEngine(
            grid_dims=grid_dims,
            piece_size=piece_size,
            num_pieces=self._num_pieces_,
            num_actions=self._num_actions_,
        )

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.append(np.zeros(self._width_ * self._height_), 0),
            high=np.append(
                np.ones(self._width_ * self._height_), self._num_pieces_ - 1
            ),
            dtype=np.int,
        )

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self._num_actions_)

    def _reset_(self) -> np.array:
        self._engine._reset()
        return self._get_obs_()

    def _step_(self, action: int) -> Tuple[np.array, float, bool, Dict[Any, Any]]:
        """
        Hard drops the current piece according to the argument provided. Terminates
        the game if a condition is met. Otherwise, a new piece is selected, and the
        anchor is reset.

        :param action: the action to be taken.
        :return: the next observation, reward, game termination indicator, and env info.
        """
        info = {}

        # Get the translation and rotation.
        translation, rotation = self._engine._all_available_actions[
            self._get_obs_()[-1]
        ][action]

        # Set the anchor and fetch the rotated piece.
        self._engine._anchor = [translation, self._piece_size_ - 1]
        self._engine._piece = self._engine._current_piece_coords[rotation]

        # Hard drop the piece and update the grid.
        self._engine._hard_drop()
        self._engine._update_grid(True)

        # Game terminates if any of the dropped piece's blocks occupies any of the
        # top piece_size rows, before any full rows are cleared.
        if np.any(self._engine._grid[:, : self._piece_size_]):
            info["num_rows_cleared"] = 0
            self._engine._final_scores = np.append(
                self._engine._final_scores, self._engine._score
            )
            return self._get_obs_(), 0.0, True, info

        # Get the reward and update the score.
        reward, num_rows_cleared = self._get_reward_()
        self._engine._score += num_rows_cleared

        # Get a new piece and update the anchor.
        self._engine._update_coords_and_anchor()

        # Update the info.
        info["num_rows_cleared"] = num_rows_cleared

        return self._get_obs_(), float(reward), False, info

    def _render_(self, mode: str) -> np.ndarray:
        return self._engine._render(mode)

    def _close_(self) -> None:
        return self._engine._close()

    def _get_obs_(self) -> np.array:
        current_grid = self._engine._grid.flatten()
        return np.append(current_grid, self._engine._current_piece_id)

    def _get_reward_(self) -> Tuple[float, int]:
        return self._engine._get_reward()


register(
    idx="simplifiedtetris-binary-v0",
    entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv",
)
