import gym
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import csv


def pickFileName():

    files = os.listdir('log/')

    return len(files)+1

class TetrisWrapper(Tetris):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 1

        f = open('log/score_count{}.txt'.format(pickFileName()), 'w+')
        self.writer = csv.writer(f)

    def reset(self):
        state = Tetris.reset(self)
        return state

    def step(self, action, /):
        obs, reward, done, info = Tetris.step(self,action)

        shaped_reward = 0

        if done:
            shaped_reward = -10
        elif reward >= 1:
            shaped_reward = reward)

        return obs, shaped_reward, done, info


    def epoch_lines(self):
         self.writer.writerow(self.score_types)
         self.score_types.clear()

