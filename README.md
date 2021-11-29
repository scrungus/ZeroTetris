<p align="center">
  <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4.gif" width="500">
</p>

<h1 align="center">Gym-SimplifiedTetris </h1>

<p align="center">
  <a href="https://www.codefactor.io/repository/github/oliveroverend/gym-simplifiedtetris">
    <img src="https://img.shields.io/codefactor/grade/github/OliverOverend/gym-simplifiedtetris?color=ff69b4&style=for-the-badge">
  </a>
  <a href="https://pypi.org/">
    <img src="https://img.shields.io/pypi/v/gym-simplifiedtetris?style=for-the-badge">
  </a>
  <a href="https://pypi.org/project/gym-simplifiedtetris/">
    <img src="https://img.shields.io/pypi/pyversions/gym-simplifiedtetris?style=for-the-badge">
  </a>
  <a href="/LICENSE.md">
    <img src="https://img.shields.io/github/license/OliverOverend/gym-simplifiedtetris?color=darkred&style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/commits/dev">
    <img src="https://img.shields.io/github/last-commit/OliverOverend/gym-simplifiedtetris/dev?style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/releases">
    <img src="https://img.shields.io/github/release-date/OliverOverend/gym-simplifiedtetris?color=teal  &style=for-the-badge">
  </a>
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues">
    <img src="https://img.shields.io/github/issues-raw/OliverOverend/gym-simplifiedtetris?color=blueviolet&style=for-the-badge">
  </a>
</p>



<p align="center">
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=bug&late=BUG_REPORT.md&title=%5BBUG%5D%3A">Report Bug
  </a>
  ·
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/issues/new?assignees=OliverOverend&labels=enhancement&late=FEATURE_REQUEST.md&title=%5BFEATURE%5D%3A">Request Feature
  </a>
  ·
  <a href="https://github.com/OliverOverend/gym-simplifiedtetris/discussions/new">Suggestions
  </a>
</p>

---

> 🟥 Simplified Tetris environments compliant with OpenAI Gym's API

Gym-SimplifiedTetris is a pip installable package that creates simplified Tetris environments compliant with [OpenAI Gym's API](https://github.com/openai/gym). The environments are simplified because the player must select the column and piece's rotation before the piece is dropped vertically downwards.  If one looks at the previous approaches to the game of Tetris, most of them use this simplified setting.

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
  - [2.3. Action and observation spaces](#23-action-and-observation-spaces)
  - [2.4. Game ending](#24-game-ending)
  - [2.5. Building more environments](#25-building-more-environments)
- [3. Agents](#3-agents)
  - [3.1. Uniform](#31-uniform)
  - [3.2. Q-learning](#32-q-learning)
  - [3.3. Heuristic](#33-heuristic)
- [4. Future work](#4-future-work)
- [5. Acknowledgements](#5-acknowledgements)
- [6. BibTeX entry](#6-bibtex-entry)
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

- numpy
- gym
- tqdm
- opencv_python
- matplotlib
- Pillow
- stable_baselines3
- dataclasses

### 1.2. Usage

The file [examples.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/examples.py) shows two examples of using an instance of the `simplifiedtetris-binary-20x10-4-v0` environment for ten game. You can create an environment using `gym.make`, supplying the environment's ID as an argument.

```python
import gym
import gym_simplifiedtetris

env = gym.make("simplifiedtetris-binary-20x10-4-v0")
obs = env.reset()

# Run 10 games of Tetris, selecting actions uniformly at random.
episode_num = 0
while episode_num < 10:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if done:
        print(f"Episode {episode_num + 1} has terminated.")
        episode_num += 1
        obs = env.reset()

env.close()
```

Alternatively, you can import the environment directly:

```python
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris

env = Tetris(grid_dims=(20, 10), piece_size=4)
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
>>> obs, reward, done, info = env.step(action)
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

### 2.3. Action and observation spaces

Each environment comes with an `observation_space` that is a `Box` space and an `action_space` t
hat is a `Discrete` space. At each time step, the agent must choose an action, an integer from a particular range.  Each action maps to a tuple that specifies the column to drop the piece and its rotation.  The number of actions available for each of the pieces is given below:

- Monominos: w
- Dominos: 2w - 1
- Trominoes: 4w - 4
- Tetriminos: 4w - 6,

where w is the grid width.  With this action space, some actions have the same effect on the grid as others.  When actions are selected uniformly at random, and the current piece is the 'O' Tetrimino, two actions are chosen with a smaller probability than the other actions.

### 2.4. Game ending

Each game terminates if any of the dropped piece's square blocks enter into the top `piece_size` rows before any full rows are cleared.  This condition ensures that scores achieved are lower bounds on the score that the agent could have obtained on a standard game of Tetris, as laid out in Colin Fahey's ['Standard Tetris' specification](https://www.colinfahey.com/tetris/tetris.html#:~:text=5.%20%22Standard%20Tetris%22%20specification).

### 2.5. Building more environments

The user can implement more custom Gym environments by ensuring that they inherit from `SimplifiedTetrisBaseEnv` and are registered in a similar way to this:

```python
>>> register(
>>>     incomplete_id=f"simplifiedtetris-binary",
>>>     entry_point=f"gym_simplifiedtetris.envs:SimplifiedTetrisBinaryEnv",
>>> )
```

## 3. Agents

### 3.1. Uniform

The uniform agent selects actions uniformly at random. See [run_uniform_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_uniform_agent.py) for an example of how to use it.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/20x10_4.gif" width="500">
</p>

### 3.2. Q-learning

Due to the curse of dimensionality, this agent struggles to learn as the grid's dimensions are increased; the size of the state-action space grows exponentially. The exploration rate parameter, epsilon, is linearly annealed over the training period.  Following the training period, the Q-learning agent selects the action with the highest state-action value.  See [run_q_learning_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_q_learning_agent.py) for an example of how to use it.

<p align="center">
    <img src="https://github.com/OliverOverend/gym-simplifiedtetris/raw/master/assets/7x4_3_q_learning.gif" width="500">
</p>

### 3.3. Heuristic

The heuristic agent selects the action with the highest heuristic score, based on the [Dellacherie feature set](https://arxiv.org/abs/1905.01652).  The heuristic score for each possible action is computed using the following heuristic, crafted by Pierre Dellacherie [Colin Fahey's website](https://colinfahey.com):

***- landing height + eroded cells - row transitions - column transitions -4 x holes - cumulative wells***

Similarly to how Colin Fahey implemented Dellacherie's agent, ties are broken by selecting the action with the largest priority.  Deviations from and to the left of the centre of the grid are rewarded, and rotations are punished.  See [run_heuristic_agent.py](https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/run_heuristic_agent.py) for an example of how to use it.

<p align="center">
    <img src="assets/20x10_4_heuristic.gif" width="500">
</p>

## 4. Future work

- Normalise the observation spaces
- Implement an action space that only permits non-terminal actions to be taken
- Implement more shaping rewards: potential-style, potential-based, dynamic potential-based, and non-potential. Optimise their weights using an optimisation algorithm.

## 5. Acknowledgements

This package utilises several methods from the [codebase](https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning) developed by andreanlay (2020) and the [codebase](https://github.com/Benjscho/gym-mdptetris) developed by Benjscho (2021).

## 6. BibTeX entry

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
