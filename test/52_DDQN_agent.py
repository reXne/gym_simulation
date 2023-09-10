import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from simulator1 import VAESimulatorEnv, VAE, VAESimulator
import matplotlib.pyplot as plt

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.layers(x)


class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch_size, device):
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=100000)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_net.layers[-1].out_features - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a minibatch from the memory
        minibatch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Convert the boolean tensor to a float tensor
        dones = dones.float()

        # Compute the Q-values for the current states
        current_q_values = self.q_net(states).gather(1, actions)

        # Compute the next actions using the main Q-network (Double DQN)
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)

        # Compute the target Q-values for the next states using the target Q-network and the selected next actions
        with torch.no_grad():
            target_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * target_q_values * (1 - dones.unsqueeze(1))

        # Compute the loss and update the Q-network
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# Training loop configurations
num_episodes = 1000
update_target_every = 100
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define VAE model architecture parameters
state_dim = 8
action_dim = 1
input_dim = state_dim + action_dim
hidden_dim = 128
latent_dim = 32
output_dim = state_dim + 1 + 1

# Instantiate an empty VAE model with the same architecture
vae_model = VAE(input_dim, hidden_dim, latent_dim, output_dim)
vae_model.load_state_dict(torch.load('../data/40_simulator/vae_model.pth'))
simulator = VAESimulator(vae_model, state_dim, action_dim, latent_dim, device)

env = VAESimulatorEnv(simulator, '../data/40_simulator/vae_initial_states_model.pth')
agent = DoubleDQNAgent(state_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch_size, device)

# Training loop
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.learn()

        episode_reward += reward
        state = next_state

        if done:
            agent.update_epsilon()
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")
            episode_rewards.append(episode_reward)
            break

    if (episode + 1) % update_target_every == 0:
        agent.update_target_net()

#plots
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards over Episodes')
plt.show()

# Save the trained model
torch.save(agent.q_net.state_dict(), "../models/ddqn_model.pt")