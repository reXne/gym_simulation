# Simple Neural Network approach

import torch
import torch.nn as nn

class SimulatorV1(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim):
        super().__init__()
        
        # Shared layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)  
        self.done_head = nn.Linear(hidden_dim, 1)  
        
    def forward(self, state_action):
        output = self.network(state_action)
        
        next_state = self.next_state_head(output)
        reward = self.reward_head(output)
        done = self.done_head(output)
        
        return next_state, reward, done



