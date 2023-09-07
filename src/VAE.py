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
