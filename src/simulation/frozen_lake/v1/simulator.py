# VAE approach

import torch
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)


class SimulatorV1(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.sequence_model = SequenceModel(state_dim  + action_dim, hidden_dim, state_dim)
        self.reward_model = RewardModel(state_dim  + action_dim, hidden_dim)
        self.continue_model = ContinueModel(state_dim  + action_dim, hidden_dim)

    def forward(self, state, action):
    
        state_next_logits = self.sequence_model(torch.cat([state, action], dim=1))

        reward_logits = self.reward_model(torch.cat([state, action], dim=1))
        
        continue_logits = self.continue_model(torch.cat([state, action], dim=1))
        
        return state_next_logits, reward_logits, continue_logits


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

    
