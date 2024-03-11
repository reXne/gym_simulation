import random
import gymnasium as gym
import torch
import numpy as np
import pickle
import os
import logging
from src.simulation.frozen_lake.v5.simulator import SimulatorV5
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

def save_tuples(tuples, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    logging.info(f"Tuples saved to {filename}")

def simulate_environment(env_name, num_episodes):
    env = gym.make(env_name)
    simulator_version = 'v5'
    model_path = f'./data/models/{env_name}/simulator_{simulator_version}.pth'
    
    num_states = 16  
    num_actions = 4  
    latent_dim=4
    hidden_dim=8
     
    model = SimulatorV5(latent_dim=latent_dim,  hidden_dim=hidden_dim, action_dim=num_actions, state_dim=num_states)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    generated_tuples = []
    total_rewards = 0
    total_dones = 0
    reward_distribution = {}
    state_distribution = {}
    
    for episode in range(num_episodes):
        state_idx = env.reset()[0]  # Assuming env.reset() returns a single integer state index
        state = to_one_hot(state_idx, num_states).unsqueeze(0)
        
        is_initial = True
        done = False
        while not done:
            action_idx = select_random_action(num_actions)
            action = to_one_hot(action_idx, num_actions).unsqueeze(0)
            state_action = torch.cat([state, action], dim=-1)
            
            with torch.no_grad():
                next_state_logits, reward_logits, done_logits,  decoder_logits, prior_dist, posterior_dist= model(state, action)
            reward_dist = Bernoulli(logits=reward_logits)
            done_dist = Bernoulli(logits=done_logits)
            
            next_state_idx = torch.argmax(next_state_logits, dim=1).item()
            
            reward = reward_dist.sample().item()
            done = done_dist.sample().item() > 0.5
            
            # Update distributions
            reward_distribution[reward] = reward_distribution.get(reward, 0) + 1
            state_distribution[next_state_idx] = state_distribution.get(next_state_idx, 0) + 1
            
            # Append tuple with indices and reward
            generated_tuples.append((state_idx, action_idx, reward, next_state_idx, done, is_initial))
            is_initial = False

            total_rewards += reward
            if not done:
                state_idx = next_state_idx  # Update for next iteration
                state = to_one_hot(state_idx, num_states).unsqueeze(0)
            else:
                total_dones += 1
                
    avg_reward = total_rewards / num_episodes
    save_path = f'./data/simulated_tuples/{env_name}/simulated_tuples_{simulator_version}.pkl'
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
    simulator_version = 'v5'
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f'./logs/03_simulate_environment_{simulator_version}.log',
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

