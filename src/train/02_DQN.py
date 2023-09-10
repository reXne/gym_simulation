import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
import wandb
#wandb login

# Hyperparameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# Initialize wandb
wandb.init(project="dqn-cartpole")

# Neural Network for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, input_dim, output_dim, use_ddqn=False):
        self.policy_net = QNetwork(input_dim, output_dim).float()
        self.target_net = QNetwork(input_dim, output_dim).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)
        self.buffer = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.use_ddqn = use_ddqn

        # Watch the policy network in wandb
        wandb.watch(self.policy_net)

    def choose_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(state)).argmax().item()
        else:
            return random.randint(0, 1)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.policy_net(state)

        if self.use_ddqn:
            next_action = self.policy_net(next_state).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state).gather(1, next_action).squeeze(1)
        else:
            next_q_values = self.target_net(next_state).max(1)[0]

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + GAMMA * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss to wandb
        wandb.log({"Loss": loss.item()})

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def get_epsilon(self):
        return self.epsilon

def train_dqn(episodes=500, use_ddqn=False, render=False, wandb_logging=False):
    env = gym.make('CartPole-v1')
    if render:
        env = gym.wrappers.Monitor(env, "videos", force=True)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, use_ddqn=use_ddqn)
    rewards = []

    score = 0
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        episode_reward = 0
        done = False       

        max_timesteps = 1000
        timesteps = 0

        while not done and timesteps < max_timesteps:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            episode_reward += reward
            state = next_state
            timesteps += 1

        rewards.append(episode_reward)
        score += episode_reward
        avg_score = score / (episode + 1)
        print(f'DQN Policy: Episode {episode}, Episode reward {episode_reward}, Running average {avg_score}')
   
        # Log the episode reward and average reward to wandb
        wandb.log({"Episode Reward": episode_reward, "Running Average": avg_score})
                # Log to Weights & Biases
        if wandb_logging:
            wandb.log({
                "Episode Reward": episode_reward,
                "Running Average": score / (episode + 1),
                "Epsilon": agent.get_epsilon()   # Log epsilon here
            })

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    env.close()
    return rewards

if __name__ == "__main__":
    episodes = 500
    dqn_rewards = train_dqn(episodes, use_ddqn=False, render=False, wandb_logging=True)
    cum_avg_rewards = np.cumsum(dqn_rewards) / (np.arange(episodes) + 1)


    # Plotting 1
    plt.plot(dqn_rewards, label='DQN', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Performance of DQN episode reward')
    plt.legend()
    plt.savefig('./data/plots/DQN.png')
    plt.show()

    # Plotting
    plt.figure(figsize=(10, 5))  # Adjusted the figure size for better clarity
    plt.plot(dqn_rewards, label='DQN Rewards', color='blue')
    plt.plot(cum_avg_rewards, label='Cumulative Avg', color='red', linestyle='dashed')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Performance of DQN cumulative episode reward')
    plt.legend()
    plt.savefig('./data/plots/DQN_with_CumAvg.png')
    plt.show()

    # Save the plot to wandb
    wandb.log({"DQN Performance": [wandb.Image(plt)]})

    # Finish the wandb run
    wandb.finish()







    # Save the plot to wandb
    wandb.log({"DQN Performance": [wandb.Image(plt)]})

    # Finish the wandb run
    wandb.finish()
