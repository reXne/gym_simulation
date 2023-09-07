import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean and log variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim + 2)  # next_observation, reward, done
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        h = self.encoder(torch.cat([state, action], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(torch.cat([z, action], dim=1))
        next_obs = out[:, :-2]
        reward = out[:, -2]
        done = torch.sigmoid(out[:, -1])
        return next_obs, reward, done, mu, logvar


def load_tuples(filename):
    with open(filename, 'rb') as f:
        tuples = pickle.load(f)
    return tuples

def train_vae(tuples, epochs=10):
    # Extract data from tuples
    old_observations, actions, rewards, observations, dones = zip(*tuples)

    # Convert data to PyTorch tensors
    old_observations_tensor = torch.FloatTensor(np.stack(old_observations))
    actions_tensor = torch.FloatTensor(actions).unsqueeze(1)
    observations_tensor = torch.FloatTensor(np.stack(observations))
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
    dones_tensor = torch.FloatTensor([float(d) for d in dones]).unsqueeze(1)

    # Initialize model and optimizer
    model = VAE(len(old_observations[0]), 1, 128, 20)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the loss
    def vae_loss(predicted_next_obs, predicted_rewards, predicted_dones, next_observations_tensor, rewards_tensor, dones_tensor, mu, logvar):
        BCE_obs = nn.MSELoss()(predicted_next_obs, next_observations_tensor)
        BCE_reward = nn.MSELoss()(predicted_rewards, rewards_tensor)
        BCE_done = nn.BCELoss()(predicted_dones.view(-1, 1), dones_tensor)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE_obs + BCE_reward + BCE_done + KLD

    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_next_obs, predicted_rewards, predicted_dones, mu, logvar = model(old_observations_tensor, actions_tensor)
        loss = vae_loss(predicted_next_obs, predicted_rewards, predicted_dones, observations_tensor, rewards_tensor, dones_tensor, mu, logvar)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model


if __name__ == "__main__":
    tuples = load_tuples(filename="./data/sampled_tuples/tuples_data.pkl")
    print(tuples[0])  # Print the first tuple to verify
    model = train_vae(tuples, epochs=10)
    # Save the model if needed
    torch.save(model.state_dict(), 'vae_model.pth')
