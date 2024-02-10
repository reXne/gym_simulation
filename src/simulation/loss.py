import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

def compute_loss(outputs, target_state, target_reward, target_done, next_state):
    # Assuming 'outputs' is unpacked here as before or within the function
    state_next_pred, reward_dist, done_dist, decoded, posterior, prior = outputs

    # VAE Loss Components
    recon_loss = F.mse_loss(decoded, target_state)
    kl_loss = kl_divergence(posterior, prior).mean()

    # Sequence Model Loss
    sequence_model_loss = F.mse_loss(state_next_pred, next_state)

    # Reward Model Loss
    reward_loss = -reward_dist.log_prob(target_reward).mean()
    
    # Done Model Loss
    done_loss = -done_dist.log_prob(target_done.float()).mean()

    # Combine the losses
    total_loss = recon_loss + kl_loss + sequence_model_loss + reward_loss + done_loss
    return total_loss, {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'sequence_model_loss': sequence_model_loss.item(),
        'reward_loss': reward_loss.item(),
        'done_loss': done_loss.item(),
    }

    
