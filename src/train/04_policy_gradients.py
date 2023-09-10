import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LR = 0.001

# Policy Network using a simple feed-forward neural network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# Policy Gradient Agent
class PGAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = PolicyNetwork(input_dim, output_dim).float()
        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)
        self.trajectory = []
        
    def choose_action(self, state):
        probs = self.policy_net(torch.FloatTensor(state))
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self):
        total_reward = sum([x[2] for x in self.trajectory])
        returns = []

        # Calculate the returns (discounted rewards)
        G = 0
        for s, a, r in reversed(self.trajectory):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        for (s, a, r), G in zip(self.trajectory, returns):
            self.optimizer.zero_grad()
            prob = self.policy_net(torch.FloatTensor(s))
            loss = -torch.log(prob[a]) * G  # negative for gradient ascent
            loss.backward()
            self.optimizer.step()

        self.trajectory = []

    def store_transition(self, state, action, reward):
        self.trajectory.append((state, action, reward))

def train_pg(episodes=500, render=False):
    env = gym.make('CartPole-v1')
    if render:
        env = gym.wrappers.Monitor(env, "videos", force=True)

    agent = PGAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        episode_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward)
            episode_reward += reward
            state = next_state

        agent.update()
        rewards.append(episode_reward)
        print(f'PG Policy: Episode {episode}, Reward: {episode_reward}')

    env.close()
    return rewards

if __name__ == "__main__":
    episodes = 1000
    pg_rewards = train_pg(episodes, render=False)

    # Plotting
    plt.plot(pg_rewards, label='Policy Gradient', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Performance of Policy Gradient')
    plt.legend()
    plt.savefig('./data/plots/PG.png')
    plt.show()
