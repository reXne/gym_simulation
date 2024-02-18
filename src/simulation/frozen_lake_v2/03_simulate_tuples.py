import random
import gymnasium as gym
import torch
import numpy as np
import pickle
import os
import logging
from simulation.cartpole.simulator import RSSM

def select_random_action(num_actions):
    """Selects a random action and returns it as a one-hot encoded tensor."""
    action_index = random.randint(0, num_actions - 1)  # Randomly select an action index
    action_tensor = to_one_hot(action_index, num_actions)
    return action_tensor

def to_one_hot(index, num_classes):
    """Converts an integer index into a one-hot encoded tensor."""
    one_hot_tensor = torch.zeros(num_classes)
    one_hot_tensor[index] = 1.0
    return one_hot_tensor

def load_model(model_path, input_dim, action_dim=1, hidden_dim=8, latent_dim=2):
    model = RSSM(input_dim, action_dim, hidden_dim, latent_dim)
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
    model_path = f'./data/models/rssm_model_{env_name}.pth'
    if env_name == 'CartPole-v1':
        model = load_model(model_path, input_dim = 4, action_dim =  1)
    if env_name == 'FrozenLake-v1':
        model = load_model(model_path, input_dim = 16, action_dim = 4)   
    
    generated_tuples = []
    total_rewards = 0
    total_dones = 0
    reward_distribution = {0: 0, 1: 0}
    
    for episode in range(num_episodes):
        # Correctly handle the observation returned from env.reset()
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]  # Assuming the first item is the observation if it's a tuple

        if env_name == 'FrozenLake-v1':
            num_states = 16
            num_actions=4
            observation = to_one_hot(observation, num_states)
            
        if env_name == 'CartPole-v1':
            num_states = 4 
            num_actions = 1  
        
        # observation = np.array(observation).reshape(1, -1)  # Ensure observation is numpy array and reshape
        observation = torch.FloatTensor(observation).unsqueeze(0)
        
        done = False
        while not done:
            action = select_random_action(num_actions).unsqueeze(0)  # For FrozenLake-v1, there are 4 possible actions
            # Since the model might expect a flat array as input, adjust accordingly
            # Flatten the action tensor if your model expects a 1D input for actions
            # action = action.unsqueeze(0)  
            with torch.no_grad():
                next_observation, reward_dist, done_dist, _, _, _ = model(observation, action)
                reward = reward_dist.sample().item()
                done = done_dist.sample().item() > 0.5

            # generated_tuples.append((observation.numpy().flatten().tolist(), action.item(), reward, next_observation.numpy().flatten().tolist(), done))
            generated_tuples.append((observation.numpy().flatten().tolist(), action.squeeze().tolist(), reward, next_observation.numpy().flatten().tolist(), done))

            reward_distribution[int(reward)] += 1
            total_rewards += reward
            if not done:
                observation = next_observation  # Update observation for the next step
            else:
                total_dones += 1

    avg_reward = total_rewards / num_episodes
    stats = {
        "average_reward": avg_reward,
        "total_episodes": num_episodes,
        "total_dones": total_dones,
        "reward_distribution": reward_distribution
    }

    save_path=f'./data/simulated_tuples/simulated_tuples_{env_name}.pkl'
    save_tuples(generated_tuples, save_path)
    return stats

def main(env_name):
    
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
    # main(env_name='CartPole-v1')
    main(env_name='FrozenLake-v1')
    # num_episodes=100
    # env_name='FrozenLake-v1'
    
