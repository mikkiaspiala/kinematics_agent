import numpy as np
import itertools
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from utils import Transition, ReplayMemory


class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 200)
        self.fc4 = nn.Linear(200, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x


class Agent(object):
    def __init__(self, state_space, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=64, gamma=0.98):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using: ", self.device)
        self.state_space_dim = state_space
        self.n_actions = n_actions
        self.policy_net = DQN(state_space, self.n_actions, hidden_size).to(self.device)
        self.target_net = DQN(state_space, self.n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].cpu().detach()

        expected_state_action_values = reward_batch + self.gamma*next_state_values.to(self.device)

        loss = F.smooth_l1_loss(state_action_values.squeeze().to(self.device),
                                expected_state_action_values.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.policy_net(state.to(self.device))
                action = torch.argmax(q_values).item()
                return action
        else:
            action = random.randrange(self.n_actions)
            return action

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

