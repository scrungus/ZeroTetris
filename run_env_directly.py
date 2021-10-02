from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

env = Tetris(
    grid_dims=(20, 10),
    piece_size=4,
)
