# dreamer approach

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)

class SimulatorV4(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder_model = EncoderModel(obs_dim, hidden_dim)
        self.decoder_model = DecoderModel(hidden_dim + latent_dim, hidden_dim, obs_dim)
        self.representation_model = RepresentationModel(hidden_dim, hidden_dim, latent_dim)
        self.dynamics_model = DynamicsModel(hidden_dim, hidden_dim, latent_dim)
        self.sequence_model = SequenceModel(hidden_dim + latent_dim + action_dim, hidden_dim, hidden_dim)
        self.reward_model = RewardModel(hidden_dim + latent_dim , hidden_dim)
        self.continue_model = ContinueModel(hidden_dim + latent_dim , hidden_dim)

    def forward(self, state, action):
        h =  self.encoder_model(state)
        mu, logvar =  self.representation_model(h)
        std = torch.exp(0.5 * logvar)
        posterior_dist = Normal(mu, std)     
        z = posterior_dist.rsample()
   
        decoder_logits = self.decoder_model(torch.cat([h, z], dim=1))  
        
        h_next = self.sequence_model(torch.cat([h, z, action], dim=1))
        mu_next, logvar_next =  self.dynamics_model(h_next)
        std_next = torch.exp(0.5 * logvar_next)
        prior_dist_next = Normal(mu_next, std_next)     
        z_next = prior_dist_next.rsample()
    
        state_next_logits =  self.decoder_model(torch.cat([h_next, z_next], dim=1)) 
        # state_next_probs =  F.softmax(state_next_logits, dim=-1)
        # state_next_sample = torch.argmax(state_next_probs, dim=-1)
        # state_next_sample_one_hot = F.one_hot(state_next_sample, num_classes=state_next_probs.shape[-1]).float()
        # state_next_sample = (state_next_sample_one_hot + state_next_probs - state_next_probs.detach()).detach()



        # acá se tiene que ocupar el next state, con cualquie acción (ej: 0)
        reward_logits = self.reward_model(torch.cat([h, z], dim=1))
        
        continue_logits = self.continue_model(torch.cat([h, z], dim=1))
        
        return state_next_logits, reward_logits, continue_logits, decoder_logits, prior_dist_next, posterior_dist

class EncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        h = self.network(x)
        return h

class RepresentationModel(nn.Module):
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
       
class DynamicsModel(nn.Module):
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        return self.network(x)  

class ContinueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        return self.network(x) 

    
class VAEInitialStateModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Assuming the state is continuous
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return recon_loss + kl_loss
