import gym

import gym_simplifiedtetris


def main():
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


if __name__ == "__main__":
    main()