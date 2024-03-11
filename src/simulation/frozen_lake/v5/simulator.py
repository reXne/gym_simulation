# VAE approach

import torch
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)


class SimulatorV5(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder_model = EncoderModel(state_dim, hidden_dim, latent_dim)
        self.decoder_model = DecoderModel(latent_dim, hidden_dim, state_dim)
        self.sequence_model = SequenceModel(state_dim + latent_dim + action_dim, hidden_dim, state_dim)


    def forward(self, state, action):
        h = self.encoder_model(state)
        mu, logvar = h
        std = torch.exp(0.5 * logvar)
        posterior_dist = Normal(mu, std)     
        z = posterior_dist.rsample()
        
        prior_dist = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        
        state_next_logits, reward_logits, continue_logits = self.sequence_model(torch.cat([state, z, action], dim=1))

        decoder_logits = self.decoder_model(z)  
        
        return state_next_logits, reward_logits, continue_logits, decoder_logits, prior_dist, posterior_dist

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
    def __init__(self, input_dim, hidden_dim, state_output_dim, reward_output_dim=1, continue_output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.state_output_layer = nn.Linear(hidden_dim, state_output_dim)
        self.reward_output_layer = nn.Linear(hidden_dim, reward_output_dim)
        self.continue_output_layer = nn.Linear(hidden_dim, continue_output_dim)
        
    def forward(self, x):
        shared_output = self.network(x)
        state_logits = self.state_output_layer(shared_output)
        reward_logits = self.reward_output_layer(shared_output)
        continue_logits = self.continue_output_layer(shared_output)
        return state_logits, reward_logits, continue_logits




