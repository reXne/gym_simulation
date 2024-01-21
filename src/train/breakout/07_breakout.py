import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
import cv2

# Hyperparameters
GAMMA = 0.99
LR = 0.00025
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_DURATION = 1000000
TARGET_UPDATE = 10000
MEMORY_CAPACITY = 1000000

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# Frame Preprocessing
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return np.array(frame, dtype=np.uint8)

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84,84), dtype=np.uint8) for _ in range(4)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=0)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=0)
    
    stacked_state = np.expand_dims(stacked_state, axis=0)
    return stacked_state, stacked_frames

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state, axis=0), action, reward, np.concatenate(next_state, axis=0), done

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, num_actions, use_ddqn=False):
        self.policy_net = DQN(input_shape, num_actions).float()
        self.target_net = DQN(input_shape, num_actions).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)
        self.buffer = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.use_ddqn = use_ddqn

    def choose_action(self, state):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(np.array(state, copy=False)).cuda()
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randint(0, 3)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(done).cuda()

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

        self.epsilon = max(EPSILON_END, self.epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY_DURATION)

# Training function
def train_breakout(episodes=5000, use_ddqn=False, render=False):
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent((4, 84, 84), env.action_space.n, use_ddqn=use_ddqn)

    stacked_frames = deque([np.zeros((84,84), dtype=np.uint8) for _ in range(4)], maxlen=4)
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        episode_reward = 0
        done = False       

        while not done:
            if render:
                env.render()

            action = agent.choose_action(state)
            next_state, reward, done, _, _  = env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        print(f'DQN Policy: Episode {episode}, Episode reward {episode_reward}, Running average {np.mean(rewards[-100:])}')

        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    env.close()
    return rewards

if __name__ == "__main__":
    episodes = 100
    breakout_rewards = train_breakout(episodes, use_ddqn=False, render=False)

    # Plotting
    plt.plot(breakout_rewards, label='DQN', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Performance of DQN on Breakout')
    plt.legend()
    plt.savefig('./data/plots/DQN_Breakout.png')
    plt.show()

