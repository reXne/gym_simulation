import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from simulation.frozen_lake.v1.simulator import SimulatorV1

import torch.nn.functional as F
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent, kl_divergence
)

def compute_loss(outputs, inputs):
    target_state, target_reward, target_done, target_state_next = inputs
    state_next_logits, reward_logits, done_logits = outputs
    
    reward_dist = Bernoulli(logits = reward_logits)
    done_dist = Bernoulli(logits = done_logits)
    target_state_next_indices = torch.argmax(target_state_next, dim=1)
    sequence_model_loss = F.cross_entropy(state_next_logits, target_state_next_indices)
    reward_loss = -reward_dist.log_prob(target_reward).mean()
    done_loss = -done_dist.log_prob(target_done.float()).mean()
    
    total_loss =  sequence_model_loss + reward_loss + done_loss
    return total_loss, {
        'total_loss': total_loss.item(),
        'sequence_model_loss': sequence_model_loss.item(),
        'reward_loss': reward_loss.item(),
        'done_loss': done_loss.item(),
    }
    
def to_one_hot(index, num_classes):
    """Converts an integer index into a one-hot encoded tensor."""
    one_hot_tensor = torch.zeros(num_classes)
    one_hot_tensor[index] = 1.0
    return one_hot_tensor

class TuplesDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

def prepare_data_loaders(dataset, batch_size=1024, split_ratio=0.8):
    """Splits the dataset into training and validation sets and prepares data loaders."""
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_and_validate(model, train_loader, val_loader, optimizer, epochs=20):
    # Initialize lists to store total losses for each epoch
    train_losses = []
    val_losses = []

    # Initialize dicts to store detailed loss components for each epoch
    train_loss_details = {'total_loss': [], 'sequence_model_loss': [], 'reward_loss': [], 'done_loss': []}
    val_loss_details = {'total_loss': [], 'sequence_model_loss': [], 'reward_loss': [], 'done_loss': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        details_accum_train = {key: 0 for key in train_loss_details.keys()}  # Accumulators for detailed components

        # Training loop
        for states, actions, rewards, next_states, dones in train_loader:
            optimizer.zero_grad()
            outputs = model(torch.cat([states, actions], dim=1))
            targets = next_states, rewards, dones, next_states
            loss, details = compute_loss(outputs, targets) 
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            for key in details:
                details_accum_train[key] += details[key]

        # Calculate and store average loss and details for the epoch
        train_losses.append(total_train_loss / len(train_loader))
        for key in train_loss_details:
            train_loss_details[key].append(details_accum_train[key] / len(train_loader))

        # Validation loop
        model.eval()
        total_val_loss = 0
        details_accum_val = {key: 0 for key in val_loss_details.keys()}  # Accumulators for detailed components

        with torch.no_grad():
            for states, actions, rewards, next_states, dones in val_loader:
                outputs = model(torch.cat([states, actions], dim=1))
                targets = next_states, rewards, dones, next_states
                val_loss, details = compute_loss(outputs, targets)  # Adjust according to actual function signature
                total_val_loss += val_loss.item()
                for key in details:
                    details_accum_val[key] += details[key]

        # Calculate and store average loss and details for the epoch
        val_losses.append(total_val_loss / len(val_loader))
        for key in val_loss_details:
            val_loss_details[key].append(details_accum_val[key] / len(val_loader))

        # Print epoch summary with consistent naming
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_losses[-1]:.4f} - Details: { {k: v[-1] for k, v in train_loss_details.items()} }")
        print(f"  Val Loss: {val_losses[-1]:.4f} - Details: { {k: v[-1] for k, v in val_loss_details.items()} }")

    
    return train_losses, val_losses, train_loss_details, val_loss_details



    
def plot_loss_components(train_losses, val_losses):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    keys = ['total_loss', 'sequence_model_loss', 'reward_loss', 'done_loss']
    
    for i, key in enumerate(keys):
        axs[i].plot(train_losses[key], label=f'Train {key}')
        axs[i].plot(val_losses[key], label=f'Validation {key}')
        axs[i].set_title(f'{key} over epochs')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig('losses.png')
    plt.show()




def main():
    # Load data
    env_name = 'FrozenLake-v1'
    simulator_version = 'v1'
    with open(f"./data/sampled_tuples/sampled_tuples_{env_name}.pkl", 'rb') as f:
        tuples = pickle.load(f)

    states, actions, rewards, next_states, dones, _ = zip(*tuples)
    states = torch.stack([to_one_hot(s, 16) for s in states])
    actions = torch.stack([to_one_hot(a, 4) for a in actions])
    rewards = torch.FloatTensor(rewards).unsqueeze(-1)
    next_states = torch.stack([to_one_hot(s, 16) for s in next_states])
    dones = torch.FloatTensor(dones).unsqueeze(-1)

    dataset = TuplesDataset(states, actions, rewards, next_states, dones)
    train_loader, val_loader = prepare_data_loaders(dataset, batch_size=1024, split_ratio=0.8)

    model = SimulatorV1(input_dim=16+4, hidden_dim=8, state_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, train_loss_details, val_loss_details = train_and_validate(model, train_loader, val_loader, optimizer, epochs=50)

    plot_loss_components(train_loss_details, val_loss_details)

    # Save the trained model
    torch.save(model.state_dict(), f'./data/models/simulator_{simulator_version}.pth')

if __name__ == "__main__":
    main()
