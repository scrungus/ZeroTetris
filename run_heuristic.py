from gym_simplifiedtetris.agents import DellacherieAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    agent = DellacherieAgent()
    env = Tetris(
        grid_dims=(20, 10),
        piece_size=4,
    )
    obs = env.reset()

    num_episodes = 0
    while num_episodes < 10:
        env.render()

        heuristic_scores = env._engine._get_dellacherie_scores()

        action = agent.predict(heuristic_scores)
        obs, rwd, done, info = env.step(action)

        if done:
            print(f"Episode {num_episodes + 1} has terminated.")
            num_episodes += 1
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
