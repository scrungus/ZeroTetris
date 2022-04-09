import gym
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import random
import numpy as np

env = Tetris(grid_dims=(10, 10), piece_size=4)
totals = []
for i in range(100):
    done = 0
    total = 0
    step = 0
    state = env.reset()
    print("Game :",i+1)
    while not done:
        action = random.randint(0,env.action_space.n-1)
        _, reward, done , _ = env.step(action)
        total += reward
    totals.append(total)

print("average over final games:",np.average(totals))



