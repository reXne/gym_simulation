import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from simulator1 import VAE, VAESimulator, VAESimulatorEnv
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

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

class DQNAgent:
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
            return random.randint(0, 1)  # Updated here to reflect the 2 possible actions (0 and 1)
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

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(self.device)  # Ensured the actions tensor has the right shape
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        current_q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg : DictConfig) -> None:
    tuples_train_file = cfg.data.files.tuples_train_file

    num_episodes = 1000
    update_target_every = 100
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    batch_size = 64
    hidden_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 54
    action_dim = 1
    latent_dim = 32
    reward_dim = 1
    done_dim = 1
    input_dim = state_dim + action_dim
    output_dim = state_dim + reward_dim + done_dim 

    vae_model = VAE(input_dim, hidden_dim, latent_dim, output_dim)
    vae_model.load_state_dict(torch.load('./models/vae_model.pth'))
    simulator = VAESimulator(vae_model, state_dim, action_dim, reward_dim, done_dim, latent_dim, device)
    env = VAESimulatorEnv(simulator, tuples_train_file)
    
    agent = DQNAgent(state_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch_size, device)

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

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

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.show()

    torch.save(agent.q_net.state_dict(), "../models/dqn_model.pt")

if __name__ == '__main__':
    main()
