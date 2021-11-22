import gym
import numpy as np
from tqdm import tqdm

from ..agents.q_learning import QLearningAgent


def train_q_learning(
    env: gym.Env, agent: QLearningAgent, num_eval_timesteps: int
) -> QLearningAgent:
    """
    Trains and evaluates a Q-learning agent on the SimplifiedTetris environment.

    :param env: the env to train the Q-learning agent on.
    :param agent: the Q-learning agent.
    :param num_eval_timesteps: the number of timesteps to evaluate for.
    """

    ep_return = 0
    returns = np.array([], dtype=int)
    done = False

    obs = env.reset()

    for _ in tqdm(range(num_eval_timesteps), desc="No. of time steps completed"):

        action = agent.predict(obs)

        next_obs, reward, done, info = env.step(action)

        agent.learn(reward=reward, obs=obs, next_obs=next_obs, action=action)
        ep_return += info["num_rows_cleared"]

        # Epsilon annealing.
        agent.epsilon -= 1 / (num_eval_timesteps)

        if done:
            obs = env.reset()
            returns = np.append(returns, ep_return)
            done = False
            ep_return = 0
        else:
            obs = next_obs

    agent.epsilon = 0

    return agent
