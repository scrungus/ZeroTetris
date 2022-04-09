import gym
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


class TetrisWrapper(Tetris):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 1

    def reset(self):
        state = Tetris.reset(self)
        return state

    def step(self, action, /):
        obs, reward, done, info = Tetris.step(self,action)

        shaped_reward = 0

        if done:
            shaped_reward = -10
        elif reward >= 1:
            shaped_reward = reward
            print("row cleared",shaped_reward)

        return obs, shaped_reward, done, info

