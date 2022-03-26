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



from stable_baselines.common.vec_env import SubprocVecEnv

import gym 
import gym_simplifiedtetris

import multiprocessing
from multiprocessing import set_start_method

# In[35]:


class CriticNet(nn.Module):
    def __init__(self, obs_size, hidden_size = 70):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        value = self.critic(x)
        return value

class ActorNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size = 50):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return dist, action


class ActorCritic():
    def __init__(self, critic, actor, hidden_size = 50):
        self.critic = critic
        self.actor = actor 
    
    @torch.no_grad()
    def __call__(self, state: torch.Tensor):
        dist, action = self.actor(state)
        probs = dist.log_prob(action)
        val = self.critic(state)
        
        return dist, action, probs, val


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
        lr: float = 3e-4,
        batch_size: int = 5,
        clip_eps: float = 0.2,
        tau : float = 0.95,
        epoch_steps: int = 1000,
        gamma: float = 0.99
    ):
        super().__init__()
        self.save_hyperparameters()
        
        def make_env():
            def _thunk():
                env = gym.make("simplifiedtetris-binary-10x10-2-v0")
                return env
            return _thunk

        envs = [make_env() for i in range(multiprocessing.cpu_count()/2)]
        print("Running Environments on ",multiprocessing.cpu_count()/2,"cores")
        self.envs = SubprocVecEnv(envs)
        self.states = self.envs.reset()
        self.eps_steps = 0
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        
        self.batch_states = []
        self.batch_actions = []
        self.batch_probs = []
        self.batch_advs = []
        self.batch_vals = []
        self.ep_rewards = []
        self.ep_vals = []
        self.epoch_rewards = []
        self.avg_reward = 0
        self.avg_ep_reward = 0
        
        self.critic = CriticNet(obs_size)
        self.actor = ActorNet(obs_size,n_actions)
        
        self.agent = ActorCritic(self.critic, self.actor)
    
    def forward(self, x):

        dist, action = self.actor(x)
        val = self.critic(x)
        
        return dist, action, val
        
    def act_loss(self,state,action,prob_old,val,adv):
        dist, _ = self.actor(state)
        prob = dist.log_prob(action)
        ratio = torch.exp(prob - prob_old)
        #PPO update
        clip = torch.clamp(ratio, 1 - self.hparams.clip_eps, 1 + self.hparams.clip_eps) * adv
        loss = -(torch.min(ratio * adv, clip)).mean()
        return loss
    
    def crit_loss(self,state,action,prob_old,val,adv):
        val_new = self.critic(state)
        #MSE
        loss = (val - val_new).pow(2).mean()
        return loss
        
    def compute_gae(self, rewards, values, next_val):
        

        rs = rewards + [next_val]
        vals = values + [next_val]
        
        x = []
        for i in range(len(rs)-1):
            x.append(rs[i]+self.hparams.gamma*vals[i+1] - vals[i])
    
        a = self.compute_reward(x, self.hparams.gamma * self.hparams.tau)

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
            actions = []
            probs = []
            vals = []
            for state in self.states:
                _, action, prob, val = self.agent(torch.Tensor(state))
                actions.append(action.item())
                probs.append(prob)
                vals.append(val.item())

            self.env.step_async(actions)
            next_states, rewards, dones, _ = self.envs.step_wait()

            self.eps_steps += len(self.states)
            
            self.batch_states.append(self.states)
            self.batch_actions.append(actions)
            self.batch_probs.append(probs)
            self.ep_rewards.append(rewards)
            self.ep_vals.append(vals)

            self.states = next_states
            
            end = i == (self.hparams.epoch_steps -1)
                
            if end:
                if end and not done:
                    #if epoch ends before terminal state, bootstrap value
                    with torch.no_grad():
                        _,_,_,val = self.agent(self.state)
                        next_val = val.item()
                else:
                    next_val = 0
                
                #compute batch discounted rewards
                self.batch_vals += self.compute_reward(self.ep_rewards,self.hparams.gamma)
                self.batch_advs += self.compute_gae(self.ep_rewards,self.ep_vals, next_val)
                
                self.epoch_rewards.append(sum(self.ep_rewards))
                self.ep_rewards.clear()
                self.ep_vals.clear()
                self.ep_step = 0
                self.state = torch.Tensor(self.env.reset())
                data = zip(self.batch_states,
                            self.batch_actions,
                            self.batch_probs,
                            self.batch_vals,
                            self.batch_advs)

                for (s, a, p, v, ad) in data:
                    yield s, a, p, v, ad
                    
                #logs
                self.avg_reward = sum(self.epoch_rewards)/self.hparams.epoch_steps
                self.avg_ep_reward = sum(self.epoch_rewards)/len(self.epoch_rewards)
                self.epoch_rewards.clear()
                
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_probs.clear()
                self.batch_vals.clear()
                self.batch_advs.clear()
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        state,action,prob_old,val,adv = batch
        # normalize adv
        adv = (adv - adv.mean())/adv.std()
        
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("avg_reward", self.avg_reward, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        
        if optimizer_idx == 0:
            loss = self.act_loss(state, action, prob_old, val, adv)
            self.log('act_loss', loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)
            return loss

        elif optimizer_idx == 1:
            loss = self.crit_loss(state, action, prob_old, val, adv)
            self.log('crit_loss', loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)
            return loss

    
    def configure_optimizers(self) -> List[Optimizer]:
        a_opt = optim.Adam(self.actor.parameters(), lr=self.hparams.lr)
        c_opt = optim.Adam(self.critic.parameters(), lr=self.hparams.lr)
        return a_opt,c_opt
    
    def __dataloader(self):
        dataset = RLDataSet(self.make_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader
    
    def train_dataloader(self):
        return self.__dataloader()


# In[38]:


model = PPOLightning()
tb_logger = TensorBoardLogger("log/")

trainer = Trainer(
    gpus=0,
    max_epochs=20000,
    logger=tb_logger
)

#trainer.fit(model)

model = PPOLightning().load_from_checkpoint("/home/scrungus/Documents/dissertation/Tetris/gym-simplifiedtetris/log/default/version_2/checkpoints/epoch=19999-step=3999999.ckpt")
print("Playing")
env = gym.make("simplifiedtetris-binary-10x10-3-v0")
state = env.reset()

done = 0
while not done:
    _,action,_ = model(torch.Tensor(state))
    state, _, done, _ = env.step(action.item())
    env.render()
