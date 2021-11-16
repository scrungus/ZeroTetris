from gym_simplifiedtetris.helpers import train_q_learning
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from gym_simplifiedtetris.agents import QLearningAgent
from gym_simplifiedtetris.helpers.eval_agent import eval_agent


def main():
    """
    Trains and evaluates a Q-learning agent.
    """
    grid_dims = (7, 4)
    env = Tetris(grid_dims=grid_dims, piece_size=3)

    q_table_dims = [2 for _ in range(grid_dims[0] * grid_dims[1])]
    q_table_dims += [env._num_pieces_] + [env._num_actions_]

    agent = QLearningAgent(q_table_dims)
    agent = train_q_learning(env=env, agent=agent, num_eval_timesteps=10000)
    eval_agent(agent=agent, env=env, num_episodes=10, render=True)


if __name__ == "__main__":
    main()
