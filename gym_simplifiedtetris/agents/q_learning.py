from typing import Optional
import numpy as np


class QLearningAgent:
    """
    A class representing a Q-learning agent.

    :param q_table_dims: the q-table dimensions.
    :param alpha: the learning rate parameter.
    :param gamma: the discount rate parameter.
    :param epsilon: the exploration rate of the epsilon-greedy policy.
    """

    def __init__(
        self,
        q_table_dims: list,
        alpha: Optional[float] = 0.2,
        gamma: Optional[float] = 0.99,
        epsilon: Optional[float] = 1.0,
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self._q_table = np.zeros((q_table_dims), dtype=float)
        self._num_actions = q_table_dims[-1]

    def predict(
        self,
        obs: np.array,
    ) -> int:
        """
        Returns an action whilst following an epsilon-greedy policy.

        :param available_actions: the actions available to the agent.
        :param obs: a NumPy array containing the observation given to the agent by the env.
        :return: an integer correspoding to the action chosen by the Q-learning agent.
        """

        # Choose an action at random with probability epsilon.
        if np.random.rand(1)[0] <= self.epsilon:
            return np.random.choice(self._num_actions)

        # Choose greedily from the set of all actions.
        return np.argmax(self._q_table[tuple(obs)])

    def learn(
        self,
        reward: float,
        obs: np.array,
        next_obs: np.array,
        action: int,
    ) -> None:
        """
        Updates the Q-learning agent's Q-table.

        :param reward: the reward given to the agent by the env after taking the action 'action'.
        :param obs: the old observation given to the agent by the env.
        :param next_obs: the next observation given to the agent by the env having taken an action.
        :param action: the action taken that generated next_obs.
        """

        # Update the Q-table.
        current_obs_action = tuple(list(obs) + [action])
        max_q_value = np.max(self._q_table[tuple(next_obs)])
        self._q_table[current_obs_action] += self.alpha * (
            reward + self.gamma * max_q_value - self._q_table[current_obs_action]
        )
