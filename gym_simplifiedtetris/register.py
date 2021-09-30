from gym.envs.registration import register as gym_register

env_list: list = []


def register(
        idx: str,
        entry_point: str,
) -> None:
    """
    This function performs some checks on the arguments provided, and then
    registers the custom environments in Gym.
    """
    assert idx.startswith(
        "simplifiedtetris-"), 'Env ID should start with "simplifiedtetris-".'
    assert entry_point.startswith(
        "gym_simplifiedtetris.envs:SimplifiedTetris"), 'Entry point should\
            start with "gym_simplifiedtetris.envs:SimplifiedTetris".'
    assert entry_point.endswith("Env"), 'Entry point should end with "Env".'
    assert idx not in env_list, f'Already registered env id: {idx}'

    gym_register(
        id=idx,
        entry_point=entry_point,
        kwargs={
            'grid_dims': (20, 10),
            'piece_size': 4,
        },
    )
    env_list.append(idx)
