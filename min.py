from torch import Tensor, nn
import torch

class ActorNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size = 64):
        super().__init__()

        print(obs_size)
        self.actor = nn.Sequential(nn.LSTM(input_size=obs_size, hidden_size=4, batch_first=True))

    def forward(self, x):
        x = torch.randn(1,4,101)
        logits = self.actor(x)
        logits = torch.nan_to_num(logits)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return dist, action


class ActorCritic():
    def __init__(self, critic, actor):
        self.critic = critic
        self.actor = actor

    @torch.no_grad()
    def __call__(self, state: torch.Tensor):
        dist, action = self.actor(state)
        probs = dist.log_prob(action)
        val = self.critic(state)

        return dist, action, probs, val

model = ActorCritic(None,ActorNet(101,20))

model(0)


