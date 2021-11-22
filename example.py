import gym

from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def run_example_1():
    env = gym.make("simplifiedtetris-binary-v0")
    obs = env.reset()

    # Run 10 games of Tetris, selecting actions uniformly at random.
    episode_num = 0
    while episode_num < 10:
        env.render()
        action = env.action_space.sample()
        obs, rwd, done, info = env.step(action)

        if done:
            print(f"Episode {episode_num + 1} has terminated.")
            episode_num += 1
            obs = env.reset()

    env.close()


def run_example_2():
    env = Tetris(grid_dims=(20, 10), piece_size=4)


def main():
    run_example_1()
    run_example_2()


if __name__ == "__main__":
    main()
