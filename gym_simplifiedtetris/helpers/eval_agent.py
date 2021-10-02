from typing import Tuple

from tqdm import tqdm
import gym
import numpy as np


def eval_agent(
    agent: object,
    env: gym.Env,
    num_eval_episodes: int,
    render: bool,
) -> Tuple[float, float]:
    """
    Evaluates the agents performance on the game of SimplifiedTetris and returns the mean score.

    :param agent: the agent to evaluate on the env.
    :param env: the env to evaluate the agent on.
    :param num_eval_episodes: the number of games to evaluate the trained agent.
    :param render: a boolean that if True renders the agent playing SimplifiedTetris after training.
    :return: the mean and std score obtained from letting the agent play num_eval_episodes games.
    """

    returns = np.zeros(num_eval_episodes)

    # Reset the game scores.
    env._engine._final_scores = np.array([], dtype=int)

    for episode_id in tqdm(range(num_eval_episodes), desc="No. of episodes completed"):

        obs = env.reset()
        done = False

        while not done:

            if render:
                env.render()

            action = agent.predict(obs)

            obs, _, done, info = env.step(action)
            returns[episode_id] += info["num_rows_cleared"]

    env.close()

    mean_score = np.mean(returns)
    std_score = np.std(returns)

    print(
        f"\nScore obtained from averaging over {num_eval_episodes} "
        f"games: {mean_score:.1f} +/- {std_score:.1f}"
    )

    return mean_score, std_score
