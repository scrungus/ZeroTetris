import numpy as np


class UniformAgent:
    """
    A class representing an agent that selects actions uniformly at random.

    :param num_actions: the number of actions available to the agent in each state.
    """

    def __init__(self, num_actions: int):
        self._num_actions = num_actions

    def predict(self) -> int:
        """
        Selects an action uniformly at random.

        :return: the action chosen by the agent.
        """
        return np.random.randint(0, self._num_actions)
