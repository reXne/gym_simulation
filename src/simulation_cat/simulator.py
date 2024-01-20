import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, latent_dim):
        super(RSSM, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # logits for latent state
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # next_observation
        )
        
        # Dynamics Predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # logits for next latent state
        )
        
        # Reward Predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Done Predictor
        self.done_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize the output weights of the reward predictor to zeros
        nn.init.zeros_(self.reward_predictor[-1].weight)
        nn.init.zeros_(self.reward_predictor[-1].bias)

    def sample_from_logits(self, logits):
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).squeeze()

    def forward(self, state, action):
        # Encoding
        logits = self.encoder(state)
        z = self.sample_from_logits(logits)
        
        # Dynamics Prediction
        next_logits = self.dynamics_predictor(torch.cat([z.unsqueeze(1), action], dim=1))
        z_next = self.sample_from_logits(next_logits)
        
        # Reward and Done Prediction from z (current latent state)
        reward = self.reward_predictor(torch.cat([z.unsqueeze(1), action], dim=1))
        done = self.done_predictor(torch.cat([z.unsqueeze(1), action], dim=1))

        # Decoding
        next_obs = self.decoder(z_next)
        
        return next_obs, reward, done, logits, next_logits
