from typing import Tuple

import numpy as np


class SimplifiedTetrisShapingReward(object):
    """
    A class representing a potential-based shaping reward.
    """

    reward_range = (-1, 5)

    def __init__(self):
        self.heuristic_range = {"min": 1000, "max": -1}

        # The old potential is 1 because there are no holes at the start of a game.
        self.old_potential = 1

        self.initial_potential = self.old_potential

    def _get_reward(self) -> Tuple[float, int]:
        """
        Gets the potential-based shaping reward.

        :return: the potential-based shaping reward and the number of lines cleared.
        """
        num_lines_cleared = self._engine._clear_rows()
        heuristic_value = self._engine._get_holes()

        if heuristic_value > self.heuristic_range["max"]:
            self.heuristic_range["max"] = heuristic_value

        if heuristic_value < self.heuristic_range["min"]:
            self.heuristic_range["min"] = heuristic_value

        new_potential = np.clip(
            1
            - (heuristic_value - self.heuristic_range["min"])
            / (self.heuristic_range["max"] + 1e-9),
            0,
            1,
        )
        shaping_reward = (new_potential - self.old_potential) + num_lines_cleared
        self.old_potential = new_potential
        return shaping_reward, num_lines_cleared

    def _get_terminal_reward(self) -> float:
        """
        Returns the terminal potential-based shaping reward.

        :return: the terminal potential-based shaping reward.
        """
        terminal_shaping_reward = -self.old_potential
        self.old_potential = self.initial_potential
        return terminal_shaping_reward
