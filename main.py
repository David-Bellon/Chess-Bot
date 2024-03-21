import chess
import chess.svg
import matplotlib.pyplot as plt
from architecture import Actor, Critic
import torch
from torch.distributions import MultivariateNormal

class PPO():
    def __init__(self, input_dim, output_dim):
        self.actor = Actor(input_dim, output_dim)
        self.critic = Critic(input_dim)
        self.variance = torch.full((output_dim, ), fill_value=0.6)

    def get_action(self, obs):
        mean = self.actor(obs)
        cov_mat = torch.diag(self.variance).unsqueeze(0)
        dist = MultivariateNormal(mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        critic = self.critic(obs)

        return action.detach(), action_logprob.detach(), critic.detach()

ppo = PPO(100, 5)
print(ppo.get_action(torch.rand(100)))