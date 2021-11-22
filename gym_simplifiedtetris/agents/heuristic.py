import numpy as np


class HeuristicAgent(object):
    """
    A class representing an agent that selects the action with the largest heurstic score.
    """

    @staticmethod
    def predict(heuristic_scores: np.array) -> int:
        """
        Returns the action that yields the largest heuristic score.

        :param heuristic_scores: the heuristic scores for each action.
        :return: the action with the largest heuristic score.
        """
        return np.argmax(heuristic_scores)
