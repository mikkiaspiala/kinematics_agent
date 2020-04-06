import simulator
import dqn_agent
import torch
import matplotlib.pyplot as plt



def train(glie=200, episodes=5000, gamma=0.98, epsilon=0.05, memory_size=10000, batch_size=32):
    env = simulator.Robot()
    state_space = 4*3+4*3+3
    action_space = 4
    movements = [-0.01, 0, 0.01]
    agent = dqn_agent.Agent(state_space, action_space, gamma, epsilon, batch_size, memory_size, movements)
    overall_cumulative_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0
        epsilon = glie/(glie+episode)

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            cumulative_reward += reward
            experience = {}
            experience['state'] = state
            experience['action'] = action
            experience['next_state'] = next_state
            experience['reward'] = reward
            experience['done'] = done
            agent.memory.push(experience)
            agent.update_network()
            state = next_state

        overall_cumulative_rewards.append(cumulative_reward)
        print("Episode: ", episode, " finished. Cumulative reward: ", cumulative_reward)
        plot_rewards(overall_cumulative_rewards)

    print('Episodes complete.')
    plt.ioff()
    plt.show()


def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too 
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated 

train()
