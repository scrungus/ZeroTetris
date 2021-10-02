from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    env = Tetris(
        grid_dims=(20, 10),
        piece_size=4,
    )


if __name__ == "__main__":
    main()
