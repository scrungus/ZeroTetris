from typing import Tuple

import numpy as np
from gym import spaces
from gym_simplifiedtetris.envs.simplified_tetris_base_env import \
    SimplifiedTetrisBaseEnv
from gym_simplifiedtetris.envs.simplified_tetris_engine import \
    SimplifiedTetrisEngine
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
            grid_dims: tuple,
            piece_size: int,
    ):
        super(SimplifiedTetrisBinaryEnv, self).__init__(
            grid_dims=grid_dims,
            piece_size=piece_size,
        )

        self.engine = SimplifiedTetrisEngine(
            grid_dims=grid_dims,
            piece_size=piece_size,
            num_pieces=self.num_pieces,
            num_actions=self.num_actions,
        )

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.append(np.zeros(self.width * self.height), 1),
            high=np.append(np.ones(self.width * self.height), self.num_pieces),
            dtype=np.int
        )

    @property
    def action_space(self):
        return spaces.Discrete(self.num_actions)

    def _reset_(self) -> np.array:
        self.engine.reset()
        return self._get_obs()

    def _step_(self, action: int) -> Tuple[np.array, float, bool, dict]:
        """
        Hard drops the current piece according to the argument provided. Terminates
        the game if a condition is met. Otherwise, a new piece is selected, and the 
        anchor is reset.

        :param action: the action to be taken.
        :return: the next observation, reward, game termination indicator, and env info.
        """
        info = {}

        # Get the translation and rotation.
        translation, rotation = self.engine.all_available_actions[self._get_obs(
        )[-1]][action]

        # Set the anchor and fetch the rotated piece.
        self.engine.anchor = [translation, self.piece_size - 1]
        self.engine.piece = self.engine.current_piece_coords[rotation]

        # Hard drop the piece and update the grid.
        self.engine.hard_drop()
        self.engine.update_grid(True)

        # Game terminates if any of the dropped piece's blocks occupies any of the
        # top piece_size rows, before any full rows are cleared.
        if np.any(self.engine.grid[:, :self.piece_size]):
            info['num_rows_cleared'] = 0
            self.engine.final_scores = np.append(
                self.engine.final_scores, self.engine.score)
            return self._get_obs(), 0.0, True, info

        # Get the reward and update the score.
        reward, num_rows_cleared = self._get_reward()
        self.engine.score += num_rows_cleared

        # Get a new piece and update the anchor.
        self.engine.update_coords_and_anchor()

        # Update the info.
        info['num_rows_cleared'] = num_rows_cleared

        return self._get_obs(), float(reward), False, info

    def _render_(self, mode: str) -> np.ndarray:
        return self.engine.render(mode)

    def _close_(self):
        return self.engine.close()

    def _get_obs(self) -> np.array:
        current_grid = np.clip(self.engine.grid.flatten(), 0, 1)
        return np.append(current_grid, self.engine.current_piece_id)

    def _get_reward(self) -> Tuple[float, int]:
        return self.engine.get_reward()


register(
    idx='simplifiedtetris-binary-v0',
    entry_point=f'gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv',
)
