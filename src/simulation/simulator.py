import torch
import torch.nn as nn

class RSSM(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, latent_dim):
        super(RSSM, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean and log variance
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # next_observation
        )
        
        # Dynamics Predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean and log variance for next latent state
        )
        
        # Reward Predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # Predict from z_next
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Add sigmoid activation
        )

        # Done Predictor
        self.done_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # Predict from z_next
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Add sigmoid activation
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        # Encoding
        h = self.encoder(torch.cat([state, action], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        
        # Dynamics Prediction
        dynamics_out = self.dynamics_predictor(torch.cat([z, action], dim=1))
        mu_next, logvar_next = torch.chunk(dynamics_out, 2, dim=1)
        z_next = self.reparameterize(mu_next, logvar_next)
        
        # Reward and Done Prediction from z_next
        reward = self.reward_predictor(z_next)
        done = self.done_predictor(z_next)
        
        # Decoding
        next_obs = self.decoder(torch.cat([z_next, action], dim=1))
        
        return next_obs, reward, done, mu, logvar, mu_next, logvar_next

    def prediction_loss(self, recon_state, recon_reward, recon_done, state, reward, done):
        state_loss = nn.MSELoss()(recon_state, state)
        reward_loss = nn.MSELoss()(recon_reward, reward)
        done_loss = nn.BCELoss()(recon_done, done)
        return state_loss + reward_loss + done_loss

    def kl_divergence(self, a, b):
        mu1, logvar1 = a['mu'], a['logvar']
        mu2, logvar2 = b['mu'], b['logvar']
        
        sigma1_sq = logvar1.exp()
        sigma2_sq = logvar2.exp()
        kl_div = 0.5 * (logvar2 - logvar1 + (sigma1_sq + (mu1 - mu2).pow(2)) / sigma2_sq - 1)
        return kl_div.mean()
    
    def dynamics_loss(self, posterior, prior):
        detached_posterior = {'mu': posterior['mu'].detach(), 'logvar': posterior['logvar'].detach()}
        
        kl_div = self.kl_divergence(detached_posterior, prior)
        return torch.clamp(kl_div, min=1.0)  # Free bits

    def representation_loss(self, posterior, prior):
        detached_prior = {'mu': prior['mu'].detach(), 'logvar': prior['logvar'].detach()}
        
        kl_div = self.kl_divergence(posterior, detached_prior)
        return torch.clamp(kl_div, min=1.0)  # Free bits

    def total_loss(self, recon_state, recon_reward, recon_done, state, reward, done, posterior, prior, beta_pred=1.0, beta_dyn=0.5, beta_rep=0.1):
        L_pred = self.prediction_loss(recon_state, recon_reward, recon_done, state, reward, done)
        L_dyn = self.dynamics_loss(posterior, prior)
        L_rep = self.representation_loss(posterior, prior)
        L = beta_pred * L_pred + beta_dyn * L_dyn + beta_rep * L_rep
        return L