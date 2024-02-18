import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.simulation.frozen_lake.simulator_v1 import SequenceModel  
from src.simulation.frozen_lake.loss_v1 import compute_loss
from torch.utils.data import Dataset, DataLoader


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
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])
        
# Updated training function to handle new model outputs and loss calculation
def train_rssm(model, dataloader, optimizer, epochs=10):
    for epoch in range(epochs):
        for state_tensor, actions_tensor, rewards_tensor, state_next_tensor, dones_tensor in dataloader:
            optimizer.zero_grad()
               
            
            # Get model outputs for the batch
            inputs = state_tensor, rewards_tensor, dones_tensor, state_next_tensor

            state_action_tensor = torch.cat([state_tensor, actions_tensor], dim=1)  # Ensure dimensions match
            outputs = model(state_action_tensor)
            # Calculate loss for the batch
            len(inputs)
            len(outputs)
            
            total_loss, loss_details = compute_loss(outputs, inputs)
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item()}, Details: {loss_details}")

    return model

def main():
    # Load data
    env_name='FrozenLake-v1'
    with open(f"./data/sampled_tuples/sampled_tuples_{env_name}.pkl", 'rb') as f:
        tuples = pickle.load(f)

    states, actions, rewards, next_states, dones, initial = zip(*tuples)

    actions_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(-1)
    rewards_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(-1)
    dones_tensor = torch.FloatTensor(np.array(dones)).unsqueeze(-1)
    
    # next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(-1)

    num_states = 16  # Total number of states in FrozenLake-v1
    num_actions = 4  # Total number of actions in FrozenLake-v1
    
    states = torch.stack([to_one_hot(s, num_states) for s in states])
    # next_states = torch.stack([to_one_hot(s, num_states) for s in next_states])
    actions = torch.stack([to_one_hot(a, num_actions) for a in actions])
    
    actions_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(-1)
    actions_tensor = actions_tensor.squeeze(-1)
        


    # Create dataset and dataloader
    dataset = TuplesDataset(states, actions_tensor, rewards_tensor, next_states, dones_tensor)
    batch_size = 1024  # Adjust as needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SequenceModel(input_dim=num_states+num_actions, hidden_dim=8, state_dim=16)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_rssm(model, dataloader, optimizer, epochs=20)

    # Save the trained model
    torch.save(model.state_dict(), f'./data/models/sequence_model_{env_name}.pth')


if __name__ == "__main__":
    # main(env_name='CartPole-v1')
    main()