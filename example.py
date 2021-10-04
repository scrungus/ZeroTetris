def run_example_2():
    """
    Usage example 2.
    """
    import gym_simplifiedtetris

    env = gym_simplifiedtetris.envs.SimplifiedTetrisBinaryEnv(
        grid_dims=(20, 10),
        piece_size=4,
    )


def run_example_1():
    """
    Usage example 1.
    """
    import gym

    import gym_simplifiedtetris

    env = gym.make("simplifiedtetris-binary-v0")
    agent = gym_simplifiedtetris.agents.UniformAgent(env._num_actions_)
    obs = env.reset()

    # Run 10 games of Tetris, selecting actions uniformly at random.
    num_episodes = 0
    while num_episodes < 10:
        env.render()
        action = agent.predict()
        obs, rwd, done, info = env.step(action)

        if done:
            print(f"Episode {num_episodes + 1} has terminated.")
            num_episodes += 1
            obs = env.reset()

    env.close()


def main():
    run_example_1()
    run_example_2()


if __name__ == "__main__":
    main()
