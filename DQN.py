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

from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events



PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
print(AVAIL_GPUS)

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
                nn.Linear(hidden_size, n_actions),
                nn.Softmax()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),
                nn.Softmax()
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
        #print("Batch Size : ", batch_size)
        if(batch_size < len(self.buffer)):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        #for bayes opt
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
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

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 300) -> None:
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

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = random.randint(0,self.env.action_space.n-1)
            #maybe with high epsilon at the start, replay buffer disproportionately fills up with pass, as pass is always a choice?
        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            #print("picked : ",action)
            action = int(action.item())
            print("action : ",action)
        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

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

        self.env = Tetris(grid_dims=(10, 10), piece_size=2)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions, depth)
        self.target_net = DQN(obs_size, n_actions, depth)


        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        #print("populating...")
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
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """

        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )

        # step through environment with agent
        #print("Agent playing")
        reward, done = self.agent.play_step(self.net, epsilon, device)
        
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }

        self.log("total_reward", torch.tensor(self.total_reward).to(device), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("reward", torch.tensor(reward).to(device), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        self.writer.writerow([self.global_step, self.total_reward, loss.unsqueeze(0).item()])

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

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


def train_model(batch_size,lr,sync_rate,replay_size,warm_start_steps,eps_last_frame,sample_size,depth):

    num_epochs = 1000

    batch_size = int(batch_size)
    sync_rate = int(sync_rate)
    replay_size = int(replay_size)
    warm_start_steps = int(warm_start_steps)
    eps_last_frame = int(eps_last_frame)
    sample_size = int(sample_size)
    depth = int(depth)

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
    )

    trainer.fit(model)
    print("F:",f)

    f.close()

    env = Tetris(grid_dims=(10, 10), piece_size=2)

    totals = []

    with torch.no_grad():
        for i in range(10):
            print("iter, ",i)
            step = 0
            done = 0
            total = 0
            state = env.reset()
            while not done and step < 100000:
                q_values = model(torch.Tensor(state))
                _, action = torch.max(q_values, dim=0)
                state, reward, done, _ = env.step(action.item())
                total += reward
                step +=1
                print("step,", step)
            totals.append(total)

    return np.average(totals)

def find_params():

    pbounds = {
        "batch_size" : (4,64),
        "lr" : (1e-5,1e-3),
        "sync_rate" : (100,1000),
        "replay_size" : (1000,50000),
        "warm_start_steps" : (500,2000),
        "eps_last_frame" : (100,1000),
        "sample_size" : (100,1000),
        "depth" : (0.6,2.4)
    }

    optimizer = BayesianOptimization(
        f = train_model,
        pbounds=pbounds,
        random_state=1,
        verbose=1
        )

    logger = JSONLogger(path="log/logsDQN.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=30,
        n_iter=200,
    )


    print("Best hyperparameters found were: ", optimizer.max)

    print("others")
    for i, res in enumerate(optimizer.res):
       print("Iteration {}: \n\t{}".format(i, res))

find_params()


