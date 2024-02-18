import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)

def compute_loss(outputs, inputs):
    target_state, target_reward, target_done, target_state_next = inputs
    state_next_logits, reward_logits, done_logits = outputs
    
    reward_dist = Bernoulli(logits = reward_logits)
    done_dist = Bernoulli(logits = done_logits)
    
    
    # Secquence model
    sequence_model_loss = F.cross_entropy(state_next_logits, target_state_next)
    
    # Reward Model Loss
    reward_loss = -reward_dist.log_prob(target_reward).mean()
    
    # Done Model Loss
    done_loss = -done_dist.log_prob(target_done.float()).mean()

    # Combine the losses
    total_loss =  sequence_model_loss + reward_loss + done_loss
    return total_loss, {
        'total_loss': total_loss.item(),
        'sequence_model_loss': sequence_model_loss.item(),
        'reward_loss': reward_loss.item(),
        'done_loss': done_loss.item(),
    }

    
