from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.distributions.kl import kl_divergence

class Encoder(nn.Module):
    def __init__(self, observation_dim, hidden_dim, latent_dim, num_categories):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_categories),
        )
        self.latent_dim = latent_dim
        self.num_categories = num_categories

    def forward(self, observation):
        logits = self.fc(observation)
        logits = logits.view(-1, self.latent_dim, self.num_categories)
        return logits


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, observation_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim),
        )

    def forward(self, z):
        reconstruction = self.fc(z)
        return reconstruction


def model_forward(encoder, decoder, observation):
    logits = encoder(observation)
    posterior = OneHotCategorical(logits=logits)
    z = posterior.sample()
    reconstruction = decoder(z)
    return reconstruction, posterior, logits



# Define and initialize your model components here
encoder = Encoder(observation_dim=..., hidden_dim=..., latent_dim=..., num_categories=...)
decoder = Decoder(latent_dim=..., hidden_dim=..., observation_dim=...)

# Given observations and rewards, you can compute the loss
observations = torch.randn(batch_size, observation_dim)  # Example observations
rewards = torch.randn(batch_size)  # Example rewards

loss = reconstruction_loss_function(encoder, decoder, observations, rewards)
