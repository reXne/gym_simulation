import random
import gym
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean and log variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim + 2)  # next_observation, reward, done
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        h = self.encoder(torch.cat([state, action], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(torch.cat([z, action], dim=1))
        next_obs = out[:, :-2]
        reward = out[:, -2]
        done = torch.sigmoid(out[:, -1])
        return next_obs, reward, done, mu, logvar

def load_model(model_path, input_dim, action_dim=1, hidden_dim=128, latent_dim=20):
    model = VAE(input_dim, action_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def simulate_environment(initial_observation, model, num_steps):
    observation = torch.FloatTensor(initial_observation).unsqueeze(0)
    for _ in range(num_steps):
        action = torch.FloatTensor([[random.choice([0, 1])]])  # Assuming binary action space
        with torch.no_grad():
            next_observation, reward, done, _, _ = model(observation, action)
        print(f"Action: {action.item()}, Reward: {reward.item()}, Done: {done.item() > 0.5}, Next Observation: {next_observation.squeeze().tolist()}")
        if done.item() > 0.5:
            break
        observation = next_observation

if __name__ == "__main__":
    # Initialize environment
    env = gym.make('CartPole-v1')
    
    # Load the trained VAE model
    model_path = 'vae_model.pth'
    model = load_model(model_path, env.observation_space.shape[0])

    # Simulate the environment using the loaded model
    initial_obs = env.reset()
    simulate_environment(initial_obs, model, 100)
