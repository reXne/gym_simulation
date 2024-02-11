import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.simulation.simulator import RSSM  # Make sure this is the updated RSSM
from src.simulation.loss import compute_loss  # Assuming the updated loss function is here

# Updated training function to handle new model outputs and loss calculation
def train_rssm(model, state_tensor, actions_tensor, state_next_tensor, rewards_tensor, dones_tensor, optimizer, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get model outputs
        outputs = model(state_tensor, actions_tensor)
        # Unpack outputs; adjust as per your model's return values
        state_next_pred, reward_dist, done_dist, decoded, posterior, prior = outputs
    
        # Calculate loss using the updated function
        total_loss, loss_details = compute_loss(outputs, state_tensor, rewards_tensor, dones_tensor, state_next_tensor)
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item()}, Details: {loss_details}")

    return model

# Main function
if __name__ == "__main__":
    # Load data
    with open("./data/sampled_tuples/sampled_tuples.pkl", 'rb') as f:
        tuples = pickle.load(f)

    # Extract data from tuples
    state, actions, rewards, state_next, dones = zip(*tuples)
    state_tensor = torch.FloatTensor(np.stack(state))
    actions_tensor = torch.FloatTensor(actions).unsqueeze(-1)  # Ensure correct shape
    state_next_tensor = torch.FloatTensor(np.stack(state_next))
    rewards_tensor = torch.FloatTensor(rewards)  # Adjust type if necessary
    rewards_tensor = rewards_tensor.unsqueeze(1)
    dones_tensor = torch.FloatTensor(dones).unsqueeze(-1)  # Ensure correct shape and type

    print("State tensor shape:", state_tensor.shape)
    print("Actions tensor shape:", actions_tensor.shape)
    print("State next tensor shape:", state_next_tensor.shape)
    print("Rewards tensor shape:", rewards_tensor.shape)
    print("Dones tensor shape:", dones_tensor.shape)

    # Initialize model and optimizer
    model = RSSM(state_dim = len(state[0]), action_dim = 1, hidden_dim = 8, latent_dim =2)  # Adjust parameters as necessary
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    
    # Train the model
    model = train_rssm(model, state_tensor, actions_tensor, state_next_tensor, rewards_tensor, dones_tensor, optimizer, epochs=1000)

    # Save the model
    torch.save(model.state_dict(), './data/models/rssm_model.pth')
