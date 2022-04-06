#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.loggers import TensorBoardLogger
import csv

from pytorch_lightning.callbacks import Callback

from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

# In[2]:




class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: int, n_actions: int, depth, hidden_size: int = 64):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()

        if depth == 2:
            self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
            )

    def forward(self, x):
        return self.net(x.float())


# In[3]:


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )


# In[4]:


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


# In[5]:
from pathlib import Path


def pickFileName():
    
    Path("log/trainingvals/").mkdir(parents=True, exist_ok=True)
    
    files = os.listdir('log/trainingvals/')
    
    return '{}.csv'.format(len(files)+1)


import random
class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float) -> int:

        if np.random.random() < epsilon:
            action = random.randint(0,self.env.action_space.n-1)
            #maybe with high epsilon at the start, replay buffer disproportionately fills up with pass, as pass is always a choice?
        else:
            state = torch.tensor(np.array([self.state]))

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            #print("picked : ",action)
            action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
    ) -> Tuple[float, bool]:

        action = self.get_action(net, epsilon)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        #print("done , ",done)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            #print("resetting")
            self.reset()
        return reward, done


# In[6]:


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self, 
        batch_size,
        lr,
        gamma,
        sync_rate,
        replay_size,
        warm_start_steps,
        eps_last_frame,
        eps_start,
        eps_end,
        sample_size,
        depth,
        writer
    ) -> None:

        self.writer = writer
        writer = -1
        super().__init__()
        self.save_hyperparameters()

        print("hparams:",self.hparams)

        self.env = Tetris(grid_dims=(10, 10), piece_size=4)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions, depth)
        self.target_net = DQN(obs_size, n_actions, depth)


        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.epoch_rewards = []
        self.avg_reward = 0
        self.ep_reward = 0
        self.done = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        print("populating...",steps)
        for i in range(steps):
            _, done = self.agent.play_step(self.net, epsilon=1.0)
            if done:
                self.env.reset()
        #print("Finished populating")
        self.env.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )

        # step through environment with agent
        reward, self.done = self.agent.play_step(self.net, epsilon)
        
        self.ep_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.done:
            self.epoch_rewards.append(self.ep_reward)
            self.ep_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())


        log = {
            "epoch_rewards": sum(self.epoch_rewards),
            "avg_reward" : self.avg_reward,
            "train_loss": loss,
        }

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch_reward", sum(self.epoch_rewards), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        status = {
            "steps": self.global_step,
            "epoch_rewards": sum(self.epoch_rewards),
            "avg_reward" : self.avg_reward,
        }

        self.writer.writerow([self.global_step, self.ep_reward, self.avg_reward])

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.sample_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()


class Resetter(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        if not pl_module.done:
            pl_module.epoch_rewards.append(pl_module.ep_reward)
        pl_module.avg_reward = sum(pl_module.epoch_rewards)/len(pl_module.epoch_rewards)

        pl_module.log("epoch_reward", sum(pl_module.epoch_rewards), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("avg_reward",pl_module.avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        pl_module.agent.reset()
        pl_module.epoch_rewards.clear()
        pl_module.ep_reward = 0


num_epochs = 25000

batch_size = 8
sync_rate = 16352
replay_size = 433020
warm_start_steps = 16352
eps_last_frame = replay_size
sample_size = 16352
depth = 2
lr = 5e-4

f = open('log/trainingvals/{}'.format(pickFileName()), 'w+')
writer = csv.writer(f)

model = DQNLightning(
        batch_size,
        lr,
        0.99, #gamma
        sync_rate,
        replay_size,
        warm_start_steps,
        eps_last_frame,
        1.0, #eps_start
        0.01, #eps_end
        sample_size,
        depth,
        writer
        )

tb_logger = TensorBoardLogger("log/")
trainer = Trainer(
        #accelerator="gpu",
        #gpus=[0],
        accelerator="cpu",
        max_epochs=num_epochs,
        val_check_interval=100,
        logger=tb_logger,
        callbacks=[Resetter()]
    )

trainer.fit(model)

f.close()

env = Tetris(grid_dims=(10, 10), piece_size=4)

totals = []

with torch.no_grad():
    for i in range(10):
        step = 0
        done = 0
        total = 0
        state = env.reset()
        while not done:
            q_values = model(torch.Tensor(state))
            _, action = torch.max(q_values, dim=0)
            state, reward, done, _ = env.step(action.item())
            total += reward
        totals.append(total)

print("average over games ",np.average(totals))

