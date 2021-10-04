import numpy as np

from gym_simplifiedtetris.agents import DellacherieAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    ep_returns = np.zeros(10)

    agent = DellacherieAgent()
    env = Tetris(
        grid_dims=(8, 10),
        piece_size=4,
    )
    obs = env.reset()

    num_episodes = 0
    while num_episodes < 10:
        env.render()

        heuristic_scores = env._engine._get_dellacherie_scores()

        action = agent.predict(heuristic_scores)
        obs, rwd, done, info = env.step(action)
        ep_returns[num_episodes] += info["num_rows_cleared"]

        if done:
            print(f"Episode {num_episodes + 1} has terminated.")
            num_episodes += 1
            obs = env.reset()

    env.close()

    print(
        f"\nScore obtained from averaging over {num_episodes} "
        f"games: {np.mean(ep_returns):.1f} +/- {np.std(ep_returns):.1f}"
    )


if __name__ == "__main__":
    main()
