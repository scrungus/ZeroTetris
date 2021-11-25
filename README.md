<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4_heuristic.gif" width="500">
</p>

<h1 align="center">Gym-SimplifiedTetris </h1>

<p align="center">
  <a href="https://www.codefactor.io/repository/github/oliveroverend/gym-simplifiedtetris">
    <img src="https://img.shields.io/codefactor/grade/github/OliverOverend/gym-simplifiedtetris?color=ff69b4&style=for-the-badge">
  </a>
  <a href="https://pypi.org/">
  <img src="https://img.shields.io/pypi/v/gym-simplifiedtetris?style=for-the-badge">
  </a>
  <a href="/LICENSE.md">
    <img src="https://img.shields.io/github/license/OliverOverend/gym-simplifiedtetris?color=red&style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/commits/dev">
    <img src="https://img.shields.io/github/last-commit/OliverOverend/gym-simplifiedtetris/dev?style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/releases">
    <img src="https://img.shields.io/github/release-date/OliverOverend/gym-simplifiedtetris?color=informational&style=for-the-badge">
  </a>
  <a href="https://github.com/XAMPPRocky/tokei#excluding-folders">
    <img src="https://img.shields.io/tokei/lines/github/OliverOverend/gym-simplifiedtetris?color=blueviolet&style=for-the-badge">
  </a>
    <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues">
    <img src="https://img.shields.io/github/issues-raw/OliverOverend/gym-simplifiedtetris?style=for-the-badge">
  </a>
</p>



<p align="center">
    <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=bug&late=BUG_REPORT.md&title=%5BBUG%5D%3A">Report Bug</a>
    ·
    <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=enhancement&late=FEATURE_REQUEST.md&title=%5BFEATURE%5D%3A">Request Feature</a>
    ·
    <a href="https://github.com/OliverOverend/gym-simplifiedtetris/discussions/new">Suggestions</a>
</p>

---

> 🟥 Research into AI using simplified Tetris environments compliant with OpenAI Gym's API

_Gym-SimplifiedTetris_ is a pip installable package capable of creating simplified Tetris environments for AI research (more specifically reinforcement learning), which are compliant with [OpenAI Gym's API](https://github.com/openai/gym). The environments are simplified because the artificial agent must select the column and piece's rotation before the piece is dropped vertically downwards. If one looks at the previous approaches to the game of Tetris, most of them use this simplified setting.

This README provides some help with the setup, gives an overview of the environments and agents and how to use them, and describes how to build more environments.

## Table of contents <!-- omit in toc -->

- [1. Getting started](#1-getting-started)
  - [1.1. Installation](#11-installation)
  - [1.2. Usage](#12-usage)
- [2. Environments](#2-environments)
  - [2.1. Available environments](#21-available-environments)
  - [2.2. Methods](#22-methods)
    - [2.2.1. `reset()` method](#221-reset-method)
    - [2.2.2. `step(action: int)` method](#222-stepaction-int-method)
    - [2.2.3. `render()` method](#223-render-method)
    - [2.2.4. `close()` method](#224-close-method)
  - [2.3. Variable dimensions and piece size](#23-variable-dimensions-and-piece-size)
  - [2.4. Action and observation spaces](#24-action-and-observation-spaces)
  - [2.5. Game ending](#25-game-ending)
  - [2.6. Building more environments](#26-building-more-environments)
- [3. Agents](#3-agents)
  - [3.1. Uniform](#31-uniform)
  - [3.2. Q-learning](#32-q-learning)
  - [3.3. Heuristic](#33-heuristic)
- [4. Future work](#4-future-work)
- [5. Acknowledgements](#5-acknowledgements)
- [6. Citing the project](#6-citing-the-project)
- [7. License](#7-license)

## 1. Getting started

### 1.1. Installation

The package is pip installable:
```bash
pip install gym-simplifiedtetris
```

Or, you can copy the repository by forking it and then download it using:

```bash
git clone https://github.com/INSERT-YOUR-USERNAME-HERE/gym-simplifiedtetris
```

Packages can be installed using pip:

```bash
cd gym-simplifiedtetris
pip install -r requirements.txt
```

Here is a list of dependencies:

- NumPy
- Gym
- OpenCV-Python
- Imageio
- Matplotlib
- Pillow
- Stable-Baselines3

### 1.2. Usage

The file [examples.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/examples.py) shows two examples of using an instance of the `simplifiedtetris-binary-v0` environment for ten games:

```python
import gym

import gym_simplifiedtetris

env = gym.make("simplifiedtetris-binary-v0")
obs = env.reset()

# Run 10 games of Tetris, selecting actions uniformly at random.
episode_num = 0
while episode_num < 10:
    env.render()
    action = env.action_space.sample()
    obs, rwd, done, info = env.step(action)

    if done:
        print(f"Episode {episode_num + 1} has terminated.")
        episode_num += 1
        obs = env.reset()

env.close()
```

Alternatively, you can import the environment directly:

```python
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

env = Tetris(
    grid_dims=(20, 10), piece_size=4
)
```

## 2. Environments

### 2.1. Available environments

There are currently 64 environments provided:

- `simplifiedtetris-binary-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid, plus the current piece's ID. A reward of +1 is given for each line cleared, and 0 otherwise
- `simplifiedtetris-partbinary-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid excluding the top `piece_size` rows, plus the current piece's ID. A reward of +1 is given for each line cleared, and 0 otherwise
- `simplifiedtetris-binary-shaped-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid, plus the current piece's ID. The reward function is a potential-based reward function based on the _holes_ feature
- `simplifiedtetris-partbinary-shaped-{height}x{width}-{piece_size}-v0`: The observation space is a flattened NumPy array containing a binary representation of the grid excluding the top `piece_size` rows, plus the current piece's ID. The reward function is a potential-based shaping reward based on the _holes_ feature

where (height, width) are either (20, 10), (10, 10), (8, 6), or (7, 4), and the piece size is either 1, 2, 3, or 4.

### 2.2. Methods

#### 2.2.1. `reset()` method

The `reset()` method returns a 1D array containing some grid binary representation, plus the current piece's ID.

```python
>>> obs = env.reset()
>>> print(obs)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 4]
```

#### 2.2.2. `step(action: int)` method

Each environment's step method returns four values:

- `observation` (**NumPy array**): a 1D array that contains some binary representation of the grid, plus the current piece's ID
- `reward` (**float**): the amount of reward received from the previous action
- `done` (**bool**): a game termination flag
- `info` (**dict**): only contains the `num_rows_cleared` due to taking the previous action

```python
>>> obs, rwd, done, info = env.step(action)
```

#### 2.2.3. `render()` method

The user has access to the following controls during rendering:

- Pause (**SPACEBAR**)
- Speed up (**RIGHT key**)
- Slow down (**LEFT key**)
- Quit (**ESC**)

```python
>>> env.render()
```

#### 2.2.4. `close()` method

The user can close all open windows using:

```python
>>> env.close()
```

### 2.3. Variable dimensions and piece size

If you are not importing the environment directly, the user can deviate from the standard grid dimensions and Tetriminos by editing the `gym_register` keyword arguments. The user can choose from four different sets of pieces: monominos, dominos, trominoes & Tetriminos. The user can select a height in the interval [`piece_size`+1, 20] and a width in the interval [`piece_size`, 10]. Below is a GIF showing games being played on a 8 x 6 grid with trominoes as the pieces.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/8x6_3.gif" width="500">
</p>

### 2.4. Action and observation spaces

Each environment comes with an `observation_space` that is a `Box` space and an `action_space` that is a `Discrete` space. At each time step, the artificial agent must choose an action (an integer from a particular range). Each action maps to a translation/rotation tuple that specifies the column to drop the piece and its rotation. The ranges for the four different piece sizes are:

- Monominos: [0, w - 1]
- Dominos: [0, 2w - 2]
- Trominoes: [0, 4w - 5]
- Tetriminos: [0, 4w  - 7]

where w is the grid width.

### 2.5. Game ending

Each game of Tetris terminates if the following condition is satisfied: any of the dropped piece's square blocks enter into the top `piece_size` rows before any full rows are cleared. This definition ensures that scores achieved are lower bounds on the score that the agent could have obtained on a standard game of Tetris, as laid out in Colin Fahey's ['Standard Tetris' specification](https://www.colinfahey.com/tetris/tetris.html#:~:text=5.%20%22Standard%20Tetris%22%20specification).

### 2.6. Building more environments

The user can implement more custom Gym environments with different observation spaces and reward functions easily. To add more environments to `gym_simplifiedtetris.register.env_list`, ensure that they inherit from `SimplifiedTetrisBaseEnv` and are registered using:

```python
>>> register(
>>>     idx='INSERT_ENV_NAME_HERE',
>>>     entry_point='gym_simplifiedtetris.envs:INSERT_ENV_CLASS_NAME_HERE',
>>> )
```

## 3. Agents

### 3.1. Uniform

The uniform agent implemented by `gym_simplifiedtetris.UniformAgent` selects actions uniformly at random. See [run_uniform_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_uniform_agent.py) for an example of how to use the uniform agent.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4.gif" width="500">
</p>

### 3.2. Q-learning

The Q-learning agent implemented by `gym_simplifiedtetris.QLearningAgent` selects the action with the highest Q-value (state-action value). The exploration rate parameter, epsilon, is linearly annealed over the training period. Note that this agent struggles to learn as the grid's dimensions are increased (the size of the state-action space becomes too large).

See [run_q_learning_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_q_learning_agent.py) for an example of how to use the Q-learning agent.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/7x4_3_q_learning.gif" width="500">
</p>

### 3.3. Heuristic

The heuristic agent implemented by `gym_simplifiedtetris.HeuristicAgent` selects the action with the highest heuristic score, based on the [Dellacherie feature set](https://arxiv.org/abs/1905.01652).

The heuristic score for each possible action is computed using the following heuristic:

***- landing height + eroded cells - row transitions - column transitions -4 x holes - cumulative wells***

See [run_heuristic_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_heuristic_agent.py) for an example of how to use the heuristic agent.

<p align="center">
    <img src="assets/20x10_4_heuristic.gif" width="500">
</p>

## 4. Future work

- Normalise the observation spaces
- Implement an action space that only allows non-terminal actions
- Implement more shaping rewards, e.g., potential-style, potential-based, dynamic potential-based, non-potential, and optimise their weights using an optimisation algorithm

## 5. Acknowledgements

This package utilises several methods from the [codebase](https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning) developed by andreanlay (2020) and the [codebase](https://github.com/Benjscho/gym-mdptetris) developed by Benjscho (2021).

## 6. Citing the project

```
@misc{gym_simplifiedtetris,
  author = {Overend, Oliver},
  title = {gym-simplifiedtetris package for OpenAI Gym},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OliverOverend/gym-simplifiedtetris}},
}
```

## 7. License

This project is licensed under the terms of the [MIT license](/LICENSE.md).
