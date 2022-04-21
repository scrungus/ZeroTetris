import gym
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import csv
import os
import numpy as np

def pickFileName():

    files = os.listdir('log/')

    return len(files)+1

class TetrisWrapper(Tetris):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.old = 1
        self.minh = 1
        self.maxh = -1
        self.score_types = [0]*4

        self.f = open('log/score_count{}.txt'.format(pickFileName()), 'w+')
        self.writer = csv.writer(self.f)

    def reset(self):
        state = Tetris.reset(self)
        self.old = 1
        self.minh = 1
        self.maxh = -1
        return state

    def step(self, action, /):
        obs, reward, done, info = Tetris.step(self,action)

        holes = self._engine._get_holes()

        self.update_min_max(holes)

        if reward == 1:
            self.score_types[0] +=1
        elif reward == 2:
            self.score_types[1] +=1
        elif reward == 3:
            self.score_types[2] +=1
        elif reward == 4:
            self.score_types[3] +=1

        if done:
            shaped_reward = -self.old
        else:
            new = np.clip( 1 - ((holes - self.minh)/(self.maxh + 1e-12)), 0, 1)
            shaped_reward = (new-self.old) + reward
            self.old = new

        return obs, shaped_reward, done, reward, info

    def update_min_max(self, curr):

        if curr > self.maxh:
            self.maxh = curr
        if curr < self.minh:
            self.minh = curr

    def epoch_lines(self):
         self.writer.writerow(self.score_types)
         self.score_types = [0]*4
         self.f.flush()


