import torch
import numpy as np
# Ensure correct import path for SequenceModel
from src.simulation.frozen_lake.simulator_v1 import SequenceModel  
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)

class ObservationSpace:
    def __init__(self, n):
        self.n = n

class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)
    
class SimulatedGymEnvironment:
    def __init__(self, model_path, state_dim, action_dim, hidden_dim=8):
        self.model = SequenceModel(input_dim=state_dim+action_dim, hidden_dim=hidden_dim, state_dim=state_dim)
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
        
        state_action_tensor = torch.cat([current_state_tensor, action_tensor], dim=1)
        
        with torch.no_grad():
            next_state_logits, reward_logits, done_logits = self.model(state_action_tensor)
            
        next_state_dist = Categorical(logits=next_state_logits)
        next_state_index = next_state_dist.sample().item()
            
        reward_dist = Bernoulli(logits=reward_logits)
        done_dist = Bernoulli(logits=done_logits)
        
        reward = reward_dist.sample().item()
        done = done_dist.sample().item() > 0.5
        
        self.current_state_index = next_state_index
        
        return next_state_index, reward, done, {}, {}



if __name__ == "__main__":
    env_name = 'FrozenLake-v1'
    model_path = f'./data/models/rssm_model_{env_name}.pth'
    state_dim = 16
    action_dim = 4
    num_episodes = 100
    
    simulated_env = SimulatedGymEnvironment(model_path, state_dim, action_dim)
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