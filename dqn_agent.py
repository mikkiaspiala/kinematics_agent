import torch
import itertools
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Memory:
    def __init__(self, batch_size=100, memory_size=1000):
        self.data = []
        self.batch_size = batch_size
        self.memory_size = memory_size

    def push(self, experience):
        self.data.append(experience)
        self.forget_if_full()

    def sample_batch(self):
        if len(self.data) < self.batch_size:
            return self.data
        else:
            sample_batch = np.random.choice(self.data, size=self.batch_size)
            return sample_batch

    def forget_if_full(self):
        if len(self.data) > self.memory_size:
            self.data.pop(0)


class Agent():
    def __init__(self, state_space, action_space, gamma, epsilon, batch_size, memory_size, movements):
        # Hyperparameters.
        self.gamma = gamma#0.98
        self.epsilon = epsilon#0.05
        self.batch_size = batch_size#100
        self.movements = movements#[-0.01, 0, 0.01]

        self.train_device = "cpu"
        self.action_space = action_space
        self.state_space = state_space
        self.policy_network = DQN(state_space, 81).double()
        self.target_network = DQN(state_space, action_space).double()
        self.memory = Memory(batch_size=self.batch_size)
        self.actions = self.get_all_possible_actions()
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), lr=5e-3)

    def get_all_possible_actions(self):
        actions = [list(i) for i in itertools.product(self.movements, repeat=self.action_space)]
        return actions

    def get_action(self, state):
        if np.random.uniform() > self.epsilon:
            state = torch.tensor(state)
            qs = self.policy_network.forward(state)
            action_id = np.argmax(qs.detach().numpy())
            action = self.map_id_to_action(action_id)
        else:
            action_id = np.random.randint(self.action_space+1)
            action = self.actions[action_id]
        return action

    def update_network(self):
        samples = self.memory.sample_batch()
        actions = [sample['action'] for sample in samples]
        rewards = [sample['reward'] for sample in samples]
        states = torch.tensor([sample['state'] for sample in samples])
        dones = [sample['done'] for sample in samples]
        next_states = torch.tensor([sample['next_state'] for sample in samples])
        next_qs = [max(self.policy_network.forward(next_state)) for next_state in next_states]
        targets = torch.tensor([rewards[i] + self.gamma*next_qs[i] if not dones[i] else rewards[i] for i in range(len(samples))])
        predicted_qs = [max(self.policy_network.forward(state)) for state in states]
        predicted_qs = torch.stack(predicted_qs)
#        print(predicted_qs.shape)
#        print(targets.shape)
        loss = F.smooth_l1_loss(predicted_qs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#        self.target_network = self.policy_network

    def map_id_to_action(self, action_id):
        action_id = np.argmax(action_id)
        action = self.actions[action_id]
        return action



