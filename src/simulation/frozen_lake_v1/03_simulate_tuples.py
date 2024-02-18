import random
import gymnasium as gym
import torch
import numpy as np
import pickle
import os
import logging
from simulation.frozen_lake_simple.simulator import SequenceModel
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)

def select_random_action(num_actions):
    """Selects a random action and returns it as a one-hot encoded tensor."""
    action_index = random.randint(0, num_actions - 1)  # Randomly select an action index
    return action_index

def to_one_hot(index, num_classes):
    """Converts an integer index into a one-hot encoded tensor."""
    one_hot_tensor = torch.zeros(num_classes)
    one_hot_tensor[index] = 1.0
    return one_hot_tensor

def load_model(model_path, input_dim, hidden_dim=8, state_dim=16):
    model = SequenceModel(input_dim=input_dim, hidden_dim=hidden_dim, state_dim=state_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def save_tuples(tuples, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    logging.info(f"Tuples saved to {filename}")

def simulate_environment(env_name, num_episodes):
    env = gym.make(env_name)
    model_path = f'./data/models/sequence_model_{env_name}.pth'
    num_states = 16  # For FrozenLake-v1
    num_actions = 4  # For FrozenLake-v1
    
    model = load_model(model_path, input_dim=num_states + num_actions, hidden_dim=8, state_dim=num_states)
    
    generated_tuples = []
    total_rewards = 0
    total_dones = 0
    reward_distribution = {}
    state_distribution = {}
    
    for episode in range(num_episodes):
        state_idx = env.reset()[0]  # Assuming env.reset() returns a single integer state index
        state = to_one_hot(state_idx, num_states).unsqueeze(0)
        
        done = False
        while not done:
            action_idx = select_random_action(num_actions)
            action = to_one_hot(action_idx, num_actions).unsqueeze(0)
            state_action = torch.cat([state, action], dim=-1)
            
            with torch.no_grad():
                next_state_logits, reward_logits, done_logits = model(state_action)
            
            reward_dist = Bernoulli(logits=reward_logits)
            done_dist = Bernoulli(logits=done_logits)
            
            next_state_dist = Categorical(logits=next_state_logits)
            next_state_idx = next_state_dist.sample().item()
            
            reward = reward_dist.sample().item()
            done = done_dist.sample().item() > 0.5
            
            # Update distributions
            reward_distribution[reward] = reward_distribution.get(reward, 0) + 1
            state_distribution[next_state_idx] = state_distribution.get(next_state_idx, 0) + 1
            
            # Append tuple with indices and reward
            generated_tuples.append((state_idx, action_idx, reward, next_state_idx, done))
            
            total_rewards += reward
            if not done:
                state_idx = next_state_idx  # Update for next iteration
                state = to_one_hot(state_idx, num_states).unsqueeze(0)
            else:
                total_dones += 1
                
    avg_reward = total_rewards / num_episodes
    save_path = f'./data/simulated_tuples/simulated_tuples_{env_name}.pkl'
    save_tuples(generated_tuples, save_path)
    
    return {
        "average_reward": avg_reward,
        "total_episodes": num_episodes,
        "total_dones": total_dones,
        "reward_distribution": reward_distribution,
        "state_distribution": state_distribution
    }


def main():
    env_name='FrozenLake-v1'
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f'./logs/03_simulate_environment_{env_name}.log',
                        filemode='w')

    
    num_episodes = 1000  # Adjust as needed
    stats = simulate_environment(env_name, num_episodes)

    logging.info("Summary Statistics for Simulated Environment:")
    logging.info(f"Average Reward: {stats['average_reward']}")
    logging.info(f"Total Episodes: {stats['total_episodes']}")
    logging.info(f"Total Dones: {stats['total_dones']}")
    logging.info(f"Reward Distribution: {stats['reward_distribution']}")

if __name__ == "__main__":
    main()

