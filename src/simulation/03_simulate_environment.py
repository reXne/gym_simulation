import random
import gym
import torch
import numpy as np
import pickle
import os
import logging
from simulator import RSSM

# Set up logging configuration to save logs to a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/03_simulate_environment.log',  # Name of the log file
                    filemode='a')  # Append mode; use 'w' for overwrite mode


def load_model(model_path, input_dim, action_dim=1, hidden_dim=64, latent_dim=8):
    model = RSSM(input_dim, action_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_tuples(tuples, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    logging.info(f"Tuples saved to {filename}")

def simulate_environment(initial_observation, model, num_steps, save_path='./data/simulated_tuples/simulated_tuples.pkl'):
    # Handle the case where the observation is a tuple
    if isinstance(initial_observation, tuple):
        initial_observation = initial_observation[0]

    observation = torch.FloatTensor(np.array(initial_observation)).unsqueeze(0)  # Convert to numpy array first
    generated_tuples = []
    
    total_rewards = 0
    total_dones = 0
    reward_distribution = {0.0: 0, 1.0: 0}  # To keep track of reward counts
    
    for step in range(num_steps):
        action = torch.FloatTensor([[random.choice([0, 1])]])  # Assuming binary action space

        with torch.no_grad():
            next_observation, reward, done, mu, logvar, mu_next, logvar_next = model(observation, action)
            reward = (reward >= 0.5).float()
            done = (done >= 0.5)
        print(f"Observation: {observation.squeeze().tolist()} Action: {action.item()}, Reward: {reward.item()}, Done: {done.item()}, Next Observation: {next_observation.squeeze().tolist()}")
        
        # Append the tuple to the list
        generated_tuples.append((observation.squeeze().tolist(), action.item(), reward.item(), next_observation.squeeze().tolist(), done.item()))
        
        # Update reward distribution
        reward_distribution[reward.item()] += 1
        
        total_rewards += reward.item()
        if done.item() > 0.5:
            total_dones += 1
            break
        observation = next_observation

    avg_reward = total_rewards / (step + 1)  # Compute average reward based on steps taken

    stats = {
        "average_reward": avg_reward,
        "total_steps": step + 1,
        "total_dones": total_dones,
        "reward_distribution": reward_distribution
    }

    # Save the generated tuples
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    save_tuples(generated_tuples, save_path)
    
    return stats

if __name__ == "__main__":
    # Initialize environment
    env = gym.make('CartPole-v1')
    
    # Load the trained RSSM model
    model_path = './data/models/rssm_model.pth'
    model = load_model(model_path, env.observation_space.shape[0])

    # Simulate the environment using the loaded model
    initial_obs = env.reset()
    stats = simulate_environment(initial_obs, model, 10000)
    
    logging.info("Summary Statistics for Simulated Environment:")
    logging.info(f"Average Reward: {stats['average_reward']}")
    logging.info(f"Total Steps: {stats['total_steps']}")
    logging.info(f"Total Dones: {stats['total_dones']}")
    logging.info(f"Reward Distribution: {stats['reward_distribution']}")
