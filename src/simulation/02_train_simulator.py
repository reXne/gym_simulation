import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from simulator import RSSM  # Assuming you have this import


# Training function
def train_rssm(model, old_observations_tensor, actions_tensor, observations_tensor, rewards_tensor, dones_tensor, optimizer, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_next_obs, raw_predicted_rewards, predicted_dones, mu, logvar, mu_next, logvar_next = model(old_observations_tensor, actions_tensor)
        
        # Create the posterior and prior dictionaries
        posterior = {'mu': mu, 'logvar': logvar}
        prior = {'mu': mu_next, 'logvar': logvar_next}
        
        loss = model.total_loss(predicted_next_obs, raw_predicted_rewards, predicted_dones, observations_tensor, rewards_tensor, dones_tensor, posterior, prior)
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model

# Main function
if __name__ == "__main__":
    # Load data
    with open("./data/sampled_tuples/sampled_tuples.pkl", 'rb') as f:
        tuples = pickle.load(f)

    # Extract data from tuples
    old_observations, actions, rewards, observations, dones = zip(*tuples)
    old_observations_tensor = torch.FloatTensor(np.stack(old_observations))
    actions_tensor = torch.FloatTensor(actions).unsqueeze(1)
    observations_tensor = torch.FloatTensor(np.stack(observations))
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
    dones_tensor = torch.FloatTensor([float(d) for d in dones]).unsqueeze(1)

    # Initialize model and optimizer
    model = RSSM(len(old_observations[0]), 1, 32, 16)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_rssm(model, old_observations_tensor, actions_tensor, observations_tensor, rewards_tensor, dones_tensor, optimizer, epochs=30)

    # Save the model if needed
    torch.save(model.state_dict(), './data/models/rssm_model.pth')
