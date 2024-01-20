from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.distributions.kl import kl_divergence

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DynamicsPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(DynamicsPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.fc(x)

class WorldModel(nn.Module):
    def __init__(self, observation_dim, latent_dim, action_dim):
        super(WorldModel, self).__init__()
        self.encoder = Encoder(observation_dim, latent_dim)
        self.dynamics = DynamicsPredictor(latent_dim, action_dim)

    def forward(self, observation, action):
        latent = self.encoder(observation)
        next_latent = self.dynamics(latent, action)
        return next_latent, latent
