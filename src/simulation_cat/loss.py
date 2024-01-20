import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

def reconstruction_loss(recon_state, raw_recon_reward, recon_done, state, reward, done, posterior_logits, prior_logits, beta_pred=1.0, beta_dyn=0.5, beta_rep=0.1):
    
    # Prediction loss components
    state_loss = nn.MSELoss()(recon_state, state)
    reward_loss = nn.MSELoss()(raw_recon_reward, reward)
    done_loss = nn.BCELoss()(recon_done, done)
    L_pred = state_loss + reward_loss + done_loss

    # Using torch.distributions.kl.kl_divergence for KL divergence
    posterior_distribution_detached = Categorical(logits=posterior_logits.detach())
    prior_distribution = Categorical(logits=prior_logits)
    kl_div_posterior = kl_divergence(posterior_distribution_detached, prior_distribution).mean()
    
    prior_distribution_detached = Categorical(logits=prior_logits.detach())
    posterior_distribution = Categorical(logits=posterior_logits)
    kl_div_prior = kl_divergence(posterior_distribution, prior_distribution_detached).mean()
    
    L_dyn = torch.clamp(kl_div_posterior, min=1.0)  # Dynamics loss with Free bits
    L_rep = torch.clamp(kl_div_prior, min=1.0)  # Representation loss with Free bits

    # Total loss
    L = beta_pred * L_pred + beta_dyn * L_dyn + beta_rep * L_rep

    return L