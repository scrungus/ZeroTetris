import gym 
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import multiprocessing

def make_env():
    def thunk():
        env = Tetris(grid_dims=(10, 10), piece_size=2)
        return env
    return thunk

if __name__ == '__main__':
    procs = int(multiprocessing.cpu_count()/2)
    print("Total Core Count :",multiprocessing.cpu_count())
    
    
    envs = [make_env() for _ in range(procs)]
    envs = SubprocVecEnv(envs)
    states = envs.reset()
    print(states.shape)
    exit()