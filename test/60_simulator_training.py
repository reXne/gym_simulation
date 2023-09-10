import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from simulator2 import VAE, Encoder, Decoder, Simulator
from your_dataset import YourDataset
import time
import logging
import pandas as pd

def prediction_loss(recon_state, recon_reward, recon_done, state, reward, done):
    state_loss = nn.MSELoss()(recon_state, state)
    reward_loss = nn.MSELoss()(recon_reward, reward)
    print("Unique values in 'done':", done.shape)
    print("Unique values in 'recon_done':", recon_done.shape)  # New line
    
    done_loss = nn.BCELoss()(recon_done, done)
    return state_loss + reward_loss + done_loss

def dynamics_loss(mu_next_pred, log_var_next_pred, mu_next, log_var_next):
    mu_next_sg = mu_next.detach()
    log_var_next_sg = log_var_next.detach()

    kl_div = torch.mean(-0.5 * (1 + log_var_next_sg - log_var_next_pred - ((mu_next_sg - mu_next_pred).pow(2) + log_var_next_sg.exp()) / log_var_next_pred.exp()))
    return max(1.0, kl_div)

def representation_loss(mu_next_pred, log_var_next_pred, mu_next, log_var_next):
    mu_next_pred_sg = mu_next_pred.detach()
    log_var_next_pred_sg = log_var_next_pred.detach()

    kl_div = torch.mean(-0.5 * (1 + log_var_next - log_var_next_pred_sg - ((mu_next - mu_next_pred_sg).pow(2) + log_var_next.exp()) / log_var_next_pred.exp()))
    return max(1.0, kl_div)

def total_loss(recon_state, recon_reward, recon_done, state, reward, done, mu_next_pred, log_var_next_pred, mu_next, log_var_next, beta_pred=1.0, beta_dyn=0.5, beta_rep=0.1):
    L_pred = prediction_loss(recon_state, recon_reward, recon_done, state, reward, done)
    L_dyn = dynamics_loss(mu_next_pred, log_var_next_pred, mu_next, log_var_next)
    L_rep = representation_loss(mu_next_pred, log_var_next_pred, mu_next, log_var_next)

    L = beta_pred * L_pred + beta_dyn * L_dyn + beta_rep * L_rep
    return L

def train_rssm(model, data_loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_training_loss = 0
    for S, A, R, Sp, D, eligibility_matrix in data_loader:
        S = S.to(device)
        A = A.to(device)
        R = R.to(device)
        Sp = Sp.to(device)
        D = D.float().to(device)  # Convert done tensor to float

        optimizer.zero_grad()
        recon_state, recon_reward, recon_done, mu, log_var, mu_next_pred, log_var_next_pred = model(S, A)
        loss = total_loss(recon_state, Sp, recon_reward, R, recon_done, D, mu, log_var, mu_next_pred, log_var_next_pred)
        loss.backward()

        # Clip gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_training_loss += loss.item()
    return total_training_loss / len(data_loader.dataset)


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, 
                filename=cfg.logs.files.simulator2_training,
                filemode='w',
                format='%(asctime)s - %(levelname)s - \n%(message)s')

    tuples_train_file = cfg.data.files.tuples_train_resampled_file
    tuples_train = pd.read_pickle(tuples_train_file)

    S, A, R, Sp, D, eligibility_matrix = tuples_train

    dataset = YourDataset(torch.tensor(S, dtype=torch.float32),
                        torch.tensor(A, dtype=torch.float32).unsqueeze(1),
                        torch.tensor(R, dtype=torch.float32).unsqueeze(1),
                        torch.tensor(Sp, dtype=torch.float32),
                      torch.tensor(D, dtype=torch.bool).unsqueeze(1),
                      torch.tensor(eligibility_matrix, dtype=torch.float32)
                      )
    
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 54
    action_dim = 1
    hidden_dim = 64
    latent_dim = 32

    model = Simulator(state_dim, action_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss = train_rssm(model, data_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), './models/simulator.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
