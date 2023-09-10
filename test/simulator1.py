import torch
import torch.nn as nn
import numpy as np
import gym
from gym import spaces
import pickle

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

class VAESimulator:
    def __init__(self, model, state_dim, action_dim, reward_dim, done_dim, latent_dim, device):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.done_dim = done_dim
        self.latent_dim = latent_dim
        self.device = device

    def generate_next_state_reward_done(self, state, action):
        with torch.no_grad():
            input_data = torch.tensor(np.concatenate([state, action]), dtype=torch.float32).to(self.device)
            mu, log_var = torch.chunk(self.model.encoder(input_data), 2, dim=-1)
            z = self.model.reparameterize(mu, log_var)
            decoded_data = self.model.decoder(z).cpu().numpy()
            Sp_predicted = decoded_data[:self.state_dim]
            reward_predicted = decoded_data[self.state_dim:self.state_dim + self.reward_dim]
            done_predicted = np.round(decoded_data[self.state_dim + self.reward_dim])
        return Sp_predicted, reward_predicted, bool(done_predicted)

class VAESimulatorEnv(gym.Env):
    def __init__(self, simulator, tuple_dataset_path, device=None):
        super(VAESimulatorEnv, self).__init__()
        self.simulator = simulator
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = simulator.state_dim
        self.action_dim = simulator.action_dim

        self.action_space = spaces.Discrete(self.action_dim)  # Adjusted for discrete actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # Load the tuples dataset
        with open(tuple_dataset_path, "rb") as f:
            self.tuple_dataset = pickle.load(f)

    def step(self, action):
        # No need to convert to array since it's a discrete action space
        next_state, reward, done = self.simulator.generate_next_state_reward_done(self.state, action)
        self.state = next_state
        return next_state, reward, done, {}

    def reset(self):
        self.state = self.sample_initial_state()
        return self.state

    def sample_initial_state(self):
        initial_states = self.tuple_dataset[0] 
        index = np.random.choice(len(initial_states))
        initial_state = initial_states[index]
        return initial_state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
