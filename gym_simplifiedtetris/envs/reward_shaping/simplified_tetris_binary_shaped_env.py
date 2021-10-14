from typing import Sequence, Tuple

import numpy as np

from gym_simplifiedtetris.envs.simplified_tetris_binary_env import (
    SimplifiedTetrisBinaryEnv,
)
from gym_simplifiedtetris.register import register


class SimplifiedTetrisBinaryShapedEnv(SimplifiedTetrisBinaryEnv):
    """
    A class representing a SimplifiedTetris env where the reward function is a
    scaled heuristic score and the obs space is the grid's binary representation
    plus the current piece's id.

    :param grid_dims: the grid's dimensions.
    :param piece_size: the size of the pieces in use.
    """

    def __init__(
        self,
        grid_dims: Sequence[int],
        piece_size: int,
    ):
        super(SimplifiedTetrisBinaryShapedEnv, self).__init__(
            grid_dims,
            piece_size,
        )
        # Set the reward and heuristic ranges.
        self.reward_range = (-1, 5)
        self.heuristic_range = {"min": 1000, "max": -1}

        # Set the old and initial potential.
        self.old_potential = 1
        self.initial_potential = self.old_potential

    def _get_reward(self) -> Tuple[float, int]:
        """
        Gets the potential-based shaping reward.

        :return: the potential-based shaping reward and the number of lines cleared.
        """
        # Get the heuristic value.
        holes = self._engine._get_num_holes()
        num_lines_cleared = self._engine._clear_lines()
        heuristic_value = holes

        # Update the heuristic range.
        if heuristic_value > self.heuristic_range["max"]:
            self.heuristic_range["max"] = heuristic_value

        if heuristic_value < self.heuristic_range["min"]:
            self.heuristic_range["min"] = heuristic_value

        # Calculate the new potential and the shaping reward.
        new_potential = np.clip(
            1
            - (heuristic_value - self.heuristic_range["min"])
            / (self.heuristic_range["max"] + 1e-9),
            0,
            1,
        )
        shaping_reward = (new_potential - self.old_potential) + num_lines_cleared

        # Update the old potential
        self.old_potential = new_potential

        return shaping_reward, num_lines_cleared

    def _get_terminal_reward(self) -> float:
        """
        Returns the terminal potential-based shaping reward.

        :return: the terminal potential-based shaping reward.
        """
        terminal_shaping_reward = -self.old_potential

        # Reset the old potential.
        self.old_potential = self.initial_potential
        return terminal_shaping_reward


register(
    idx="simplifiedtetris-binary-shaped-v0",
    entry_point="gym_simplifiedtetris.envs:SimplifiedTetrisBinaryShapedEnv",
)
