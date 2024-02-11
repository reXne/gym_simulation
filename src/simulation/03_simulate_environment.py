import random
import gymnasium as gym
import torch
import numpy as np
import pickle
import os
import logging
from src.simulation.simulator import RSSM

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/03_simulate_environment.log',
                    filemode='w')

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

def simulate_environment(env, model, num_episodes, save_path='./data/simulated_tuples/simulated_tuples.pkl'):
    generated_tuples = []
    total_rewards = 0
    total_dones = 0
    reward_distribution = {0: 0, 1: 0}
    
    for episode in range(num_episodes):
        observation, _ = env.reset()  # Extract observation and ignore the dictionary
        observation = np.array(observation).reshape(1, -1)  # Ensure observation is numpy array and reshape
        observation = torch.FloatTensor(observation)  # Convert to tensor
        
        done = False
        while not done:
            action = torch.FloatTensor([[random.choice([0, 1])]])
            with torch.no_grad():
                next_observation, reward_dist, done_dist, _, _, _ = model(observation, action)
                reward = reward_dist.sample().item()
                done = done_dist.sample().item() > 0.5

            next_observation = next_observation.reshape(1, -1)  # Ensure next_observation is correctly formatted
            
            generated_tuples.append((observation.numpy().flatten().tolist(), action.item(), reward, next_observation.numpy().flatten().tolist(), done))
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

    # Ensure directory exists before saving
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    save_tuples(generated_tuples, save_path)
    
    return stats

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model_path = './data/models/rssm_model.pth'
    model = load_model(model_path, env.observation_space.shape[0])

    num_episodes = 1000  # Adjust as needed
    stats = simulate_environment(env, model, num_episodes)

    logging.info("Summary Statistics for Simulated Environment:")
    logging.info(f"Average Reward: {stats['average_reward']}")
    logging.info(f"Total Episodes: {stats['total_episodes']}")
    logging.info(f"Total Dones: {stats['total_dones']}")
    logging.info(f"Reward Distribution: {stats['reward_distribution']}")
