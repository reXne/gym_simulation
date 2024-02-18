# Dreamer V3 approach

import torch
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super(RSSM, self).__init__()
        self.encoder_model = EncoderModel(state_dim, hidden_dim, latent_dim)
        self.decoder_model = DecoderModel(latent_dim, hidden_dim, state_dim)
        self.sequence_model = SequenceModel(state_dim + latent_dim + action_dim, hidden_dim, state_dim)
        self.reward_model = RewardModel(state_dim + latent_dim, hidden_dim, 1)
        self.continue_model = ContinueModel(state_dim + latent_dim, hidden_dim, 1)

    def forward(self, state, action):
        h = self.encoder_model(state)
        mu, logvar = h
        std = torch.exp(0.5 * logvar)
        posterior_dist = Normal(mu, std)     
        z = posterior_dist.rsample()
        
        prior_dist = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        
        state_next_logits = self.sequence_model(torch.cat([state, z, action], dim=1))
        
        decoded_logits = self.decoder_model(z)  

        reward_logits = self.reward_model(torch.cat([state, z], dim=1))
        continue_logits = self.continue_model(torch.cat([state, z], dim=1))

        return state_next_logits, reward_logits, continue_logits, decoded_logits, prior_dist, posterior_dist

class EncoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar= nn.Linear(hidden_dim, latent_dim)  

    def forward(self, x):
        h = self.network(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class DecoderModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        logits = self.network(z)
        return logits

class SequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        logits = self.network(x)
        return logits  

class ContinueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        logits = self.network(x)
        return logits 

    
