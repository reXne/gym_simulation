import torch
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder_model = EncoderModel(state_dim, hidden_dim, latent_dim)
        self.decoder_model = DecoderModel(latent_dim, hidden_dim, state_dim)
        self.sequence_model = SequenceModel(state_dim + latent_dim + action_dim, hidden_dim, state_dim)
        self.reward_model = RewardModel(state_dim + latent_dim, hidden_dim)
        self.continue_model = ContinueModel(state_dim + latent_dim, hidden_dim)

    def forward(self, state, action, is_initial_state):

        prior = Normal(torch.zeros_like(mu), torch.ones_like(logvar))

        state_next = self.sequence_model(torch.cat([state, z, action], dim=1))

        h_next = self.encoder(state_next)
        mu_next, logvar_next = torch.chunk(h_next, 2, dim=1)
        z_next, _ = self.reparameterize(mu_next, logvar_next)

        decoded = self.decoder(z)

        reward_dist = self.reward_predictor(torch.cat([state_next, z_next], dim=1))
        done_dist = self.done_predictor(torch.cat([state_next, z_next], dim=1))

        return state_next, reward_dist, done_dist, decoded, posterior, prior

class EncoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )

    def forward(self, x):
        h = self.network(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        posterior_dist = Normal(mu, std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior

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
        return self.network(z)

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

encoded predicted next state
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )

    def forward(self, x):
        h = self.network(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        pior_dist = Normal(mu, std)
        prior = pior_dist.rsample()
        return pior_dist, prior
    
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x =  self.network(x)
        if self.output_dim==1:
            dist = Bernoulli(logits=x)
        else:
            dist = OneHotCategorical(logits=x)
        dist_independent = Independent(dist, 1)
        return dist

class ContinueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x =  self.network(x)
        dist = Bernoulli(logits=x)
        dist_independent = Independent(dist, 1)
        return dist
