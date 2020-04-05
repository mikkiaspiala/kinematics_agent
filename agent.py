import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, 64)
        self.fc2 = torch.nn.Linear(64, 100)
        self.fc3 = torch.nn.Linear(100, 64)
        self.fc4_mean = torch.nn.Linear(64, action_space)
        self.sigma = torch.nn.Parameter(torch.tensor([10.0]))
        self.init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

    def update_variance(self, k):
        c = 0.0005
        first_sigma = torch.tensor([10.0])
        self.sigma = first_sigma*np.exp(-c*k)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc4_mean(x)
        sigma = self.sigma
        output = Normal(mu, sigma)
        return output


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        baseline=20
        #discounted_rewards = discount_rewards(rewards, self.gamma)-baseline
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)


#        weighted_probs = torch.tensor([-action * reward for action in action_probs for reward in discounted_rewards])
#        weighted_probs = -action_probs * discounted_rewards[:, None]
#        print(action_probs.shape)
#        print(discounted_rewards.shape)
#        weighted_probs = -action_probs * discounted_rewards
        weighted_probs = -action_probs * discounted_rewards.reshape((-1, 1))
        loss = torch.mean(weighted_probs)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        #self.policy.update_variance(episode_number)

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        aprob = self.policy.forward(x)
        if evaluation:
#            action = torch.argmax(aprob.mean)
            action = aprob.mean
        else:
            action = aprob.sample()
        act_log_prob = aprob.log_prob(action)
        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))


