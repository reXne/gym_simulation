import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import pickle

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# Define the VAE loss function
def vae_loss(x, x_recon, mu, log_var):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div

# Train the VAE on your data
def train_vae(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for S, _ in data_loader:
        S = S.to(device)
        optimizer.zero_grad()
        S_recon, mu, log_var = model(S)
        loss = vae_loss(S, S_recon, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

class YourDataset(Dataset):
    def __init__(self, S):
        self.S = S

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.S[idx], self.S[idx]
    
def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tuples = pd.read_pickle('../data/04_tuples/tuples.pkl')

    S, A, R, Sp, D = tuples

    state_dim = S.shape[1]

    hidden_dim = 128
    latent_dim = 32

    # Train a VAE on the starting states
    starting_states = S.copy()

    vae_initial_states = VAE(state_dim, hidden_dim, latent_dim, state_dim).to(device)

    optimizer_initial_states = optim.Adam(vae_initial_states.parameters(), lr=1e-3)

    starting_states_dataset = YourDataset(torch.tensor(starting_states, dtype=torch.float32))

    starting_states_data_loader = DataLoader(starting_states_dataset, batch_size=64, shuffle=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_vae(vae_initial_states, starting_states_data_loader, optimizer_initial_states, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Save the trained VAE model for initial state distribution
    torch.save(vae_initial_states.state_dict(), '../data/40_simulator/vae_initial_states_model.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
