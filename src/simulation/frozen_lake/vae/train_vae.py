import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.frozen_lake.v2.simulator import SimulatorV2
from torch.utils.data import Dataset, DataLoader, random_split

import torch.nn.functional as F
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent, kl_divergence
)

import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    
class VAEInitialStateModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) 
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

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.fc_mu.out_features)
        generated_states = self.decoder(z)
        return generated_states


def to_one_hot(index, num_classes):
    """Converts an integer index into a one-hot encoded tensor."""
    one_hot_tensor = torch.zeros(num_classes)
    one_hot_tensor[index] = 1.0
    return one_hot_tensor

def vae_loss(recon_x, mu, logvar, x):
    targets_indices = torch.argmax(x, dim=1)  
    recon_loss = F.cross_entropy(recon_x, targets_indices, reduction='sum')
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

class StatesDataset(torch.utils.data.Dataset):
    def __init__(self, states):
        self.states = states
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx]

def prepare_data_loaders(dataset, batch_size=64, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def plot_loss_components(epochs_range, train_hist, val_hist, title):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_hist, label='Training')
    plt.plot(epochs_range, val_hist, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    env_name = 'FrozenLake-v1'
    simulator_version = 'vae'
    with open(f"./data/sampled_tuples/sampled_tuples_{env_name}.pkl", 'rb') as f:
        tuples = pickle.load(f)

    states, _, _, _, _, _ = zip(*tuples)
    states = torch.stack([to_one_hot(s, 16) for s in states])

    state_dim = 16
    hidden_dim = 8
    latent_dim = 4
    epochs = 20  


    dataset = StatesDataset(states)
    train_loader, val_loader = prepare_data_loaders(dataset, batch_size=1024, split_ratio=0.8)
    
    model = VAEInitialStateModel(state_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    early_stopping =  EarlyStopping(patience=5, min_delta=0.01, restore_best_weights=True)

    train_loss_hist, val_loss_hist = [], []
    train_recon_hist, val_recon_hist = [], []
    train_kl_hist, val_kl_hist = [], []

    for epoch in range(1, epochs + 1):
        # Initialize loss accumulators for the epoch
        epoch_train_loss, epoch_train_recon, epoch_train_kl = 0.0, 0.0, 0.0
        epoch_val_loss, epoch_val_recon, epoch_val_kl = 0.0, 0.0, 0.0
            
        model.train()
        epoch_train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, mu, logvar, data)
            loss.backward()
            epoch_train_loss += loss.item()
            optimizer.step()
            epoch_train_recon += recon_loss.item()
            epoch_train_kl += kl_loss.item()

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                recon_batch, mu, logvar = model(data)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, mu, logvar, data)
                epoch_val_loss += loss.item()
                epoch_val_recon += recon_loss.item()
                epoch_val_kl += kl_loss.item()
         
        # Average losses and record for plotting
        train_loss_hist.append(epoch_train_loss / len(train_loader.dataset))
        val_loss_hist.append(epoch_val_loss / len(val_loader.dataset))
        train_recon_hist.append(epoch_train_recon / len(train_loader.dataset))
        val_recon_hist.append(epoch_val_recon / len(val_loader.dataset))
        train_kl_hist.append(epoch_train_kl / len(train_loader.dataset))
        val_kl_hist.append(epoch_val_kl / len(val_loader.dataset))
    
        print(f'Epoch: {epoch}')
        print(f'Training Total Loss: {epoch_train_loss / len(train_loader.dataset):.4f}, '
              f'Reconstruction Loss: {epoch_train_recon / len(train_loader.dataset):.4f}, '
              f'KL Divergence: {epoch_train_kl / len(train_loader.dataset):.4f}')
        print(f'Validation Total Loss: {epoch_val_loss / len(val_loader.dataset):.4f}, '
              f'Reconstruction Loss: {epoch_val_recon / len(val_loader.dataset):.4f}, '
              f'KL Divergence: {epoch_val_kl / len(val_loader.dataset):.4f}')

        stop = early_stopping(model, epoch_val_loss)
        if stop:
            print("Early stopping triggered. Restoring best model weights.")
            model.load_state_dict(early_stopping.best_model)
            break

    torch.save(model.state_dict(), f'./data/models/vae_initial_state_model_{simulator_version}.pth')

    epochs_range = range(1, epoch + 1)
    plot_loss_components(epochs_range, train_loss_hist, val_loss_hist, 'Total VAE Loss')
    plot_loss_components(epochs_range, train_recon_hist, val_recon_hist, 'Reconstruction Loss')
    plot_loss_components(epochs_range, train_kl_hist, val_kl_hist, 'KL Divergence')
    
if __name__ == "__main__":
    main()
    