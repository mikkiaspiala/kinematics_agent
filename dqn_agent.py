import torch
import torch.nn.functional as F
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


class Memory():
    def __init__(self, batch_size=100):
        data = []

    def push(self, experience):
        data.append(experience)

    def sample_batch(self):
        if len(data) < batch_size:
            return data
        else:
            sample_batch = np.random.choice(data, size=batch_size)
            return sample_batch


class Agent(object):
    def __init__(self, brains, state_space, action_space):
        self.gamma = 0.98
        self.batch_size = 100

        self.train_device = "gpu"
        self.action_space = action_space
        self.state_space = state_space
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.policy_network = DQN(state_space, action_space)
        self.target_network = DQN(state_space, action_space)
        self.memory = Memory(batch_size=self.batch_size)

    def get_action(self, state, epsilon=0.05):
        if np.random.uniform() > epsilon:
            action = self.policy_network.forward(state)
        else:
            action = np.randint(self.action_space+1)
        return action

    def update_network(self):
        samples = self.memory.sample_batch()
        actions = [sample['action'] for sample in samples]
        rewards = [sample['reward'] for sample in samples]
        states = [sample['state'] for sample in samples]
        dones = [sample['done'] for sample in samples]
        next_states = [sample['next_state'] for sample in samples]
        next_qs = [self.policy_network.forward(next_state) for next_state in next_states]
        targets = [rewards[i] + self.gamma*next_qs[i] if not dones[i] else rewards[i] for i in range(len(samples))]
        for state, target in zip(states, target):
            prediction = self.policy_network.forward(state)
            loss = F.smooth_l1_loss(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.target_network = self.policy_network
