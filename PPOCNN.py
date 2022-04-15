#!/usr/bin/env python
# coding: utf-8

# In[34]:


from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import IterableDataset
from torch.distributions import Categorical
import gym 
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
import numpy as np

from TetrisWrapperNorm import TetrisWrapper

from pytorch_lightning.callbacks import Callback
import multiprocessing

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# In[35]:

class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size = 256):
        super().__init__()

        self.net = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16,kernel_size=8,stride=4,padding=7),
        nn.ReLU(inplace=False),
        nn.BatchNorm1d(16),
        nn.Conv1d(in_channels=16, out_channels=32,kernel_size=4,stride=1,padding=2),
        nn.ReLU(inplace=False),
        nn.BatchNorm1d(32),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(896, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            )

        self.critic = nn.Sequential(
            nn.Linear(896, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        if x.dim() == 1:
            convs = self.net(x[None,...][None,...])
        else:
            convs = self.net(x)
        convs = convs.view(convs.shape[0],-1)
        logits = self.policy_head(convs)
        logits = torch.nan_to_num(logits)
        dist = Categorical(logits=logits)
        action = dist.sample()

        value = self.critic(convs)

        with torch.no_grad():
            prob = dist.log_prob(action)

        return dist, action, prob, value


# In[36]:


class RLDataSet(IterableDataset):
    def __init__(self, batch_maker):
        self.batch_maker = batch_maker
    def __iter__(self):
        return self.batch_maker()


# In[37]:

class PPOLightning(LightningModule):
    
    def __init__(
        self,
        alr,
        clr,
        batch_size,
        clip_eps,
        lamb ,
        epoch_steps,
        gamma,
        depth,
        writer
    ):
        self.writer = writer
        writer = -1
        super().__init__()
        self.save_hyperparameters()

        print("hparams:",self.hparams)

        self.env = TetrisWrapper(grid_dims=(10, 10), piece_size=4)
        self.state = torch.Tensor(self.env.reset())
        self.ep_step = 0
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        print("actions",n_actions)
        
        self.batch_states = []
        self.batch_actions = []
        self.batch_probs = []
        self.batch_advs = []
        self.batch_vals = []
        self.ep_rewards = []
        self.ep_rewards_all = []
        self.ep_vals = []
        self.epoch_rewards = []
        self.avg_reward = 0
        self.avg_ep_reward = 0
        self.last_ep_logged = 0
        
        self.agent = ActorCritic(obs_size,n_actions)
    
    def forward(self, x):
        
        dist, action = self.actor(x)
        val = self.critic(x)
        
        return dist, action, val
        
    def loss(self,state,action,val,prob_old,adv):

        x = state[:,None,:]
        dist, _, _, val_new = self.agent(x)

        prob = dist.log_prob(action)

        ratio = torch.exp(prob - prob_old)
        #PPO update
        clip = torch.clamp(ratio, 1 - self.hparams.clip_eps, 1 + self.hparams.clip_eps) * adv
        #negative gradient descent - gradient ascent

        act_loss = -(torch.min(ratio * adv, clip)).mean()
        crit_loss = (val - val_new).pow(2).mean()

        loss = act_loss + 0.5*crit_loss
        return loss
    
        
    def compute_gae(self, rewards, values, next_val):
        

        rs = rewards
        vals = values + [next_val]
        
        x = []
        for i in range(len(rs)-1):
            x.append(rs[i]+self.hparams.gamma*vals[i+1] - vals[i])
    
        a = self.compute_reward(x, self.hparams.gamma * self.hparams.lamb)

        return a
    
    def compute_reward(self,rewards, gamma):
        
        rs = []
        sum_rs = 0

        for r in reversed(rewards):
            sum_rs = (sum_rs * gamma) + r
            rs.append(sum_rs)


        return list(reversed(rs))

    
    def make_batch(self):
        for i in range(self.hparams.epoch_steps):

            dist, action, probs, val = self.agent(self.state)

            next_state, reward, done, _ = self.env.step(action.item())
            self.ep_step += 1

            #print(self.ep_step, reward)
            
            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_probs.append(probs)
            self.ep_rewards.append(reward)
            self.ep_vals.append(val.item())

            self.state = torch.Tensor(next_state)
            
            end = i == (self.hparams.epoch_steps -1)

            if done or end:
                
                if end and not done:
                    #if epoch ends before terminal state, bootstrap value
                    with torch.no_grad():
                        #print("epoch ended early")
                        _,_,_,val = self.agent(self.state)
                        next_val = val.item()
                else:
                    next_val = 0
                
                #compute batch discounted rewards
                self.ep_rewards.append(next_val)
                self.batch_vals += self.compute_reward(self.ep_rewards,self.hparams.gamma)[:-1]
                self.batch_advs += self.compute_gae(self.ep_rewards,self.ep_vals, next_val)
                
                self.epoch_rewards.append(sum(self.ep_rewards))
                #print("Total for Ep :",sum(self.ep_rewards))
                self.ep_rewards_all.append(sum(self.ep_rewards))
                self.ep_rewards.clear()
                self.ep_vals.clear()
                self.ep_step = 0
                self.state = torch.Tensor(self.env.reset())
                
            if end:
                data = zip(self.batch_states,
                            self.batch_actions,
                            self.batch_probs,
                            self.batch_vals,
                            self.batch_advs)

                for (s, a, p, v, ad) in data:
                    yield s, a, p, v, ad
                    
                #logs
                self.avg_ep_reward = sum(self.epoch_rewards)/len(self.epoch_rewards)
                self.epoch_rewards.clear()
                
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_probs.clear()
                self.batch_vals.clear()
                self.batch_advs.clear()
    
    def training_step(self, batch, batch_idx):
        
        state,action,prob_old,val,adv = batch

        # normalize adv
        adv = (adv - adv.mean())/adv.std()
        
        for i in range(self.last_ep_logged,len(self.ep_rewards_all)):
             self.log("ep_reward",self.ep_rewards_all[i],prog_bar=True, on_step=False, on_epoch=True, logger=True)
             self.last_ep_logged += 1

        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("epoch_rewards", sum(self.epoch_rewards), prog_bar=True, on_step=False, on_epoch=True, logger=True)

        loss = self.loss(state, action, val, prob_old, adv)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True,logger=True)

        self.writer.writerow([self.global_step, self.avg_ep_reward, loss.unsqueeze(0).item()])

        return loss

    
    def configure_optimizers(self) -> Optimizer:
        opt = optim.Adam(self.agent.parameters(), lr=self.hparams.alr)
        return opt
    
    def __dataloader(self):
        dataset = RLDataSet(self.make_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader
    
    def train_dataloader(self):
        return self.__dataloader()


# In[38]:

class ReturnCallback(Callback):
    def __init__(self ):
        self.total = []

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.env.epoch_lines()

    def get_total(self):
        return self.total

from pathlib import Path
import csv
import os

def pickFileName():

    Path("log/trainingvalsPPO/").mkdir(parents=True, exist_ok=True)

    files = os.listdir('log/trainingvalsPPO/')

    return '{}.csv'.format(len(files)+1)

num_epochs=25000


f = open('log/trainingvalsPPO/{}'.format(pickFileName()), 'w+')
writer = csv.writer(f)

model = PPOLightning(
        6.99e-4,#alr,
        7.07e-4,#clr,
        80,#batch_size,
        0.208,#clip_eps,
        0.953,#lamb,
        2048, #epoch steps
        0.99, #gamma
        2,#depth,
        writer
    )

tb_logger = TensorBoardLogger("log/")

trainer = Trainer(
        accelerator="cpu",
        max_epochs=num_epochs,
        logger=tb_logger,
        callbacks=[ReturnCallback()])



trainer.fit(model)

print("finished training")

f.close()

totals = []

env = TetrisWrapper(grid_dims=(10, 10), piece_size=4)

with torch.no_grad():
    for i in range(10):
        done = 0
        total = 0
        step = 0
        state = env.reset()
        while not done:
            _,action,_ = model(torch.Tensor(state))
            state, reward, done, _ = env.step(action.item())
            total += reward
            #   print("stepped",action.item(),done)
        totals.append(total)

print("average over final games:",np.average(totals))
