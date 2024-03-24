import torch
import numpy as np
# Ensure correct import path for SequenceModel
from src.simulation.frozen_lake.v2.simulator import SimulatorV2  
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)

def to_one_hot(index, num_classes):
    """Converts an integer index into a one-hot encoded tensor."""
    one_hot_tensor = torch.zeros(num_classes)
    one_hot_tensor[index] = 1.0
    return one_hot_tensor

class ObservationSpace:
    def __init__(self, n):
        self.n = n

class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)
    
class SimulatedGymEnvironment:
    def __init__(self, model_path, state_dim, action_dim, hidden_dim, latent_dim):
        self.model = SimulatorV2(state_dim=state_dim, action_dim = action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.observation_space = ObservationSpace(state_dim)
        self.action_space = ActionSpace(action_dim)
        
        self.current_state_index = None 
        self.reset()

    def reset(self):
        # Example: Reset to a zero state or select a random initial state
        self.current_state_index = 0
        return self.current_state_index
    
    def close(self):
        pass   
    
    def step(self, action_index):
        # Convert action index to one-hot tensor
        current_state_tensor = torch.zeros(1, self.state_dim)
        current_state_tensor[0, self.current_state_index] = 1.0
        action_tensor = torch.zeros(1, self.action_dim)
        action_tensor[0, action_index] = 1.0
        
        with torch.no_grad():
            next_state_logits, _, _, _, _, _ = self.model(current_state_tensor, action_tensor)
            
        next_state_dist = Categorical(logits=next_state_logits)
        next_state_idx = next_state_dist.sample().item()      
        # next_state_idx = torch.argmax(next_state_logits).item()
        next_state = to_one_hot(next_state_idx, self.state_dim).unsqueeze(0)          
        
        with torch.no_grad():
            _, reward_logits, done_logits, _, _, _ = self.model(next_state, action_tensor)

        reward_dist = Bernoulli(logits=reward_logits)
        done_dist = Bernoulli(logits=done_logits)
        
        reward = reward_dist.sample().item()
        done = done_dist.sample().item() > 0.5
        
        self.current_state_index = next_state_idx
        
        return next_state_idx, reward, done, {}, {}



if __name__ == "__main__":
    env_name = 'FrozenLake-v1'
    simulator_version = 'v2'
    model_path = f'./data/models/{env_name}/simulator_{simulator_version}.pth'
    state_dim = 16
    action_dim = 4
    num_episodes = 100
    hidden_dim=8
    latent_dim = 8
    simulated_env = SimulatedGymEnvironment(model_path=model_path, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    total_reward = 0  # Track total reward across all episodes
    
    for episode in range(num_episodes):
        state = simulated_env.reset()
        done = False
        episode_reward = 0  # Reset episode reward
        
        while not done:
            action = np.random.choice(action_dim)  # Random action, replace with your policy
            next_state, reward, done, _, _ = simulated_env.step(action)
            episode_reward += reward  # Accumulate reward for the current episode
            state = next_state
        
        total_reward += episode_reward  # Add episode reward to total reward
        print(f"Episode: {episode+1}, Total Reward: {episode_reward}")
    
    average_reward = total_reward / num_episodes  # Calculate average reward per episode
    print(f"Average Reward per Episode: {average_reward}")