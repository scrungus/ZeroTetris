from gym_simplifiedtetris.agents import QLearningAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from gym_simplifiedtetris.helpers import train_q_learning
from gym_simplifiedtetris.helpers.eval_agent import eval_agent


def main():
    """
    Trains and evaluates a Q-learning agent.
    """
    grid_dims = (7, 4)
    env = Tetris(grid_dims=grid_dims, piece_size=4)
    agent = QLearningAgent(
        grid_dims=grid_dims, num_pieces=env._num_pieces_, num_actions=env._num_actions_
    )
    agent = train_q_learning(env=env, agent=agent, num_eval_timesteps=1000)
    eval_agent(agent=agent, env=env, num_episodes=30, render=True)


if __name__ == "__main__":
    main()
