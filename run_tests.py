import gym
from stable_baselines3.common.env_checker import check_env

from gym_simplifiedtetris.agents import UniformAgent
from gym_simplifiedtetris.register import env_list


def main() -> None:
    """
    Checks if each env created conforms to the OpenAI Gym API.
    The first observation is printed out for visual inspection.
    Ten games are played using an agent that selects actions uniformly at random.
    In every game, the reward received is validated and the env is rendered for visual inspection.
    """
    num_envs = len(env_list)

    for env_id, env_name in enumerate(env_list):
        print(f"\nTesting the env: {env_name} ({env_id+1}/{num_envs})")

        env = gym.make(env_name)
        check_env(env)

        obs = env.reset()
        print(f"First observation given: {obs}")

        agent = UniformAgent(env._num_actions_)

        num_episodes = 0
        while num_episodes < 10:
            env.render()

            action = agent.predict()

            obs, reward, done, _ = env.step(action)

            assert (
                env.reward_range[0] <= reward <= env.reward_range[1]
            ), f"Reward seen: {reward}"

            if done:
                num_episodes += 1
                obs = env.reset()

        env.close()


if __name__ == "__main__":
    main()
