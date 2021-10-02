from typing import Dict

import numpy as np


class DellacherieAgent:
    """
    A class representing Dellacherie's agent.

    :param n_actions: the maximum number of actions available to the agent.
    """

    def __init__(self):
        self.weights = np.array([-4.0, -1.0, -1.0, -1.0, -1.0, 1.0])

    def predict(self, obs: np.array) -> int:
        """
        Returns the action that yields the largest heuristic score.

        :param heuristic_scores: the heuristic scores for each action.
        :return: the action with the largest heuristic score.
        """
        heuristic_scores: Dict[float, int]

        highest_score = max(list(heuristic_scores.keys()))
        return heuristic_scores[highest_score]
