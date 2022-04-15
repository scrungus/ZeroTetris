import gym
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import csv
import os

def pickFileName():

    files = os.listdir('log/')

    return len(files)+1

class TetrisWrapper(Tetris):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 1
        self.score_types = [0]*4
        self.f = open('log/score_count{}.txt'.format(pickFileName()), 'w+')
        self.writer = csv.writer(self.f)

    def reset(self):
        state = Tetris.reset(self)
        return state

    def step(self, action, /):
        obs, reward, done, info = Tetris.step(self,action)

        shaped_reward = 0

        if reward == 1:
            self.score_types[0] +=1
        elif reward == 2:
            self.score_types[1] +=1
        elif reward == 3:
            self.score_types[2] +=1
        elif reward == 4:
            self.score_types[3] +=1


        return obs, reward, done, info


    def epoch_lines(self):
         self.writer.writerow(self.score_types)
         self.f.flush()
         self.score_types = [0]*4

