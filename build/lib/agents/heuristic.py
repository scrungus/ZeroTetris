from typing import Dict

import numpy as np


class DellacherieAgent:
    """
    A class representing Pierre Dellacherie's agent.
    """

    def predict(self, heuristic_scores: np.array) -> int:
        """
        Returns the action that yields the largest heuristic score.

        :param heuristic_scores: the heuristic scores for each action.
        :return: the action with the largest heuristic score.
        """
        best_action = np.argmax(heuristic_scores)
        return best_action
