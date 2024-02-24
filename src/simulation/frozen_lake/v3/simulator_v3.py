# VAE approach
import torch
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)
import torch.nn.functional as F


class SimulatorV3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder_model = EncoderModel(state_dim, hidden_dim, latent_dim)
        self.decoder_model = DecoderModel(latent_dim, hidden_dim, state_dim)
        self.sequence_model = SequenceModel(state_dim + latent_dim + action_dim, hidden_dim, state_dim)
        self.reward_model = RewardModel(state_dim + latent_dim, hidden_dim)
        self.continue_model = ContinueModel(state_dim + latent_dim, hidden_dim)

    def forward(self, state, action):
        h = self.encoder_model(state)
        mu, logvar = h
        std = torch.exp(0.5 * logvar)
        posterior_logits = mu, std
        posterior_dist = Normal(posterior_logits)     
        z_posterior = posterior_dist.rsample()
        
        state_next_logits = self.sequence_model(torch.cat([state, z_posterior, action], dim=1))
        state_next_probs = F.softmax(state_next_logits, dim=1)
        num_classes = state_next_logits.size(1)
        state_next =  F.one_hot(state_next_probs, num_classes=num_classes)
        
        h_next = self.encoder_model(state_next)
        mu_next, logvar_next = h_next
        std_next = torch.exp(0.5 * logvar_next)
        prior_logits_next = mu_next, std_next
        prior_dist_next = Normal(prior_logits_next)     
        z_prior_next = prior_dist_next.rsample()
        
        decoder_logits = self.decoder_model(posterior_dist)  
        reward_logits = self.reward_model(torch.cat([state_next, z_prior_next], dim=1))
        continue_logits = self.continue_model(torch.cat([state_next, z_prior_next], dim=1))

        return state_next_logits, reward_logits, continue_logits, decoder_logits, posterior_logits, prior_logits_next

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

    
