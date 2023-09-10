import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu, log_var = self.fc2(x).chunk(2, dim=-1)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        recon_state = self.fc2(x)
        return recon_state

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state):
        mu, log_var = self.encoder(state)
        z = self.reparameterize(mu, log_var)
        recon_state = self.decoder(z)
        return recon_state, mu, log_var
    
class DynamicsPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super(DynamicsPredictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class RewardPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(RewardPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.network(z)

class ContinuePredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(ContinuePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.network(z)

class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super(RSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(state_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, state_dim)
        self.vae = VAE(state_dim, hidden_dim, latent_dim)
        self.dynamics_predictor = DynamicsPredictor(latent_dim, action_dim, hidden_dim)
        self.reward_predictor = RewardPredictor(latent_dim, hidden_dim)
        self.continue_predictor = ContinuePredictor(latent_dim, hidden_dim)

    def forward(self, state, action):
        mu, log_var = self.encoder(state)
        z = self.vae.reparameterize(mu, log_var)

        mu_next_pred, log_var_next_pred = self.dynamics_predictor(torch.cat([z, action], dim=-1))
        z_next_pred = self.vae.reparameterize(mu_next_pred, log_var_next_pred)

        recon_state = self.decoder(z_next_pred)
        recon_reward = self.reward_predictor(z_next_pred)
        recon_done = self.continue_predictor(z_next_pred)

        return recon_state, recon_reward, recon_done, mu, log_var, mu_next_pred, log_var_next_pred

class Simulator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super(Simulator, self).__init__()
        self.rssm_model = RSSM(state_dim, action_dim, hidden_dim, latent_dim)

    def forward(self, state, action):
        return self.rssm_model(state, action)
    
    def generate_samples(self, state, action):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            recon_state, recon_reward, recon_done, _, _, _, _ = self.rssm_model(state_tensor, action_tensor)
            done_predicted = np.round(recon_done.cpu().numpy())
        return recon_state.cpu().numpy().squeeze(), recon_reward.cpu().numpy().squeeze(), bool(done_predicted)

class SimulatorEnv(gym.Env):
    def __init__(self, simulator_model):
        super(SimulatorEnv, self).__init__()

        self.simulator_model = simulator_model

        self.action_space = spaces.Discrete(2)  # Update this according to your action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.state = None

    def step(self, action):
        # Generate the next state, reward, and done signal using the simulator model
        self.state, reward, done = self.simulator_model.generate_samples(self.state, action)

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(4,))  # Update this according to your initial state distribution
        return self.state

    def render(self, mode='human'):
        pass
