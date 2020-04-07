import simulator
import dqn_agent
import torch
import matplotlib.pyplot as plt



def train(glie=500, episodes=50000, gamma=0.98, epsilon=0.05, memory_size=10, batch_size=4):
    env = simulator.Robot()
    n_actions = 81
    state_space = 6
    movements = [-0.01, 0, 0.01]
    TARGET_UPDATE = 20
    hidden = 12
    agent = dqn_agent.Agent(state_space, n_actions)
    overall_cumulative_rewards = []


    for episode in range(episodes):
        state = env.reset()
        done = False
        cumulative_reward = 0
        epsilon = glie/(glie+episode)

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action)
            cumulative_reward += reward
            agent.store_transition(state, action, next_state, reward, done)
            agent.update_network()
            state = next_state

        overall_cumulative_rewards.append(cumulative_reward)
        print("Episode: ", episode, " finished. Cumulative reward: ", cumulative_reward)
        plot_rewards(overall_cumulative_rewards)

        if episode % 10 == 0:
            print("Episode goal: ", env.goal, " Final position: ", env.get_robot_position()[-1]['xyz'])
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        if episode % 1000 == 0:
            torch.save(agent.policy_net.state_dict(),
                       "weights_%s_%d.mdl" % ("kinematics_agent", episode))

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
