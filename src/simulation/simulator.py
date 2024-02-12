import torch
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) 
        )
        
        # Sequence Model
        self.sequence_model = nn.Sequential(
            nn.Linear(state_dim + latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  
        )
      
        # Reward Predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) #,nn.Sigmoid()
        )
        
        # Done Predictor
        self.done_predictor = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) #,nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        q_z = Normal(mu, std)     
        z = q_z.rsample()
        return z, q_z
    
    def forward(self, state, action):
        h = self.encoder(state)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z, posterior = self.reparameterize(mu, logvar)
        
        prior = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        
        state_next = self.sequence_model(torch.cat([state, z, action], dim=1))
        
        # z_next para predecr next reward y continues
        h_next = self.encoder(state_next)
        mu_next, logvar_next = torch.chunk(h_next, 2, dim=1)
        z_next, posteior_next = self.reparameterize(mu_next, logvar_next)
        
        decoded = self.decoder(z)
        
        # Reward and Done Prediction from z (current latent state)
        reward_logits = self.reward_predictor(torch.cat([state_next, z_next], dim=1))
        reward_dist = Bernoulli(logits=reward_logits)
        reward_dist_independent = Independent(reward_dist, 1)

        done_logits = self.done_predictor(torch.cat([state_next, z_next], dim=1))
        done_dist = Bernoulli(logits=done_logits)
        done_dist_independent = Independent(done_dist, 1)

        return state_next, reward_dist, done_dist, decoded, posterior, prior

