import gym
from stable_baselines3.common.env_checker import check_env

import gym_simplifiedtetris


def main():
    """
    This function checks if each env created conforms to the OpenAI Gym API. 
    The first observation is printed out for visual inspection. Ten games are
    played using an agent that selects actions uniformly at random. In every game,
    the reward received is validated and the env is rendered for visual inspection.
    """

    num_envs = len(gym_simplifiedtetris.register.env_list)

    for env_id, env_name in enumerate(gym_simplifiedtetris.register.env_list):
        print(
            f'\nTesting the env: {env_name} ({env_id+1}/{num_envs})')

        env = gym.make(env_name)

        # Check the env conforms to the OpenAI Gym API.
        check_env(env)

        obs = env.reset()
        print(f'First observation given: {obs}')

        num_episodes = 0
        while num_episodes < 10:

            env.render()

            obs, reward, done, _ = env.step(env.action_space.sample())

            # Checks the reward is valid.
            assert env.REWARD_RANGE[0] <= reward <= env.REWARD_RANGE[
                1], f"Reward seen: {reward}"

            if done:
                num_episodes += 1
                obs = env.reset()

        env.close()


if __name__ == '__main__':
    main()
