import random
import gym
import torch
import numpy as np
import pickle
import os

from simulator import RSSM

def load_model(model_path, input_dim, action_dim=1, hidden_dim=128, latent_dim=20):
    model = RSSM(input_dim, action_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_tuples(tuples, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    print(f"Tuples saved to {filename}")

def simulate_environment(initial_observation, model, num_steps, save_path='./data/simulated_tuples/simulated_tuples.pkl'):
    # Handle the case where the observation is a tuple
    if isinstance(initial_observation, tuple):
        initial_observation = initial_observation[0]

    observation = torch.FloatTensor(np.array(initial_observation)).unsqueeze(0)  # Convert to numpy array first
    generated_tuples = []
    for _ in range(num_steps):
        action = torch.FloatTensor([[random.choice([0, 1])]])  # Assuming binary action space

        with torch.no_grad():
            next_observation, reward, done, mu, logvar, mu_next, logvar_next = model(observation, action)
            reward = (reward >= 0.5).float()

        print(f"Action: {action.item()}, Reward: {reward.item()}, Done: {done.item() > 0.5}, Next Observation: {next_observation.squeeze().tolist()}")
        
        # Append the tuple to the list
        generated_tuples.append((observation.squeeze().tolist(), action.item(), reward.item(), next_observation.squeeze().tolist(), done.item() > 0.5))
        
        if done.item() > 0.5:
            break
        observation = next_observation

    # Save the generated tuples
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    save_tuples(generated_tuples, save_path)

if __name__ == "__main__":
    # Initialize environment
    env = gym.make('CartPole-v1')
    
    # Load the trained RSSM model
    model_path = './data/models/rssm_model.pth'
    model = load_model(model_path, env.observation_space.shape[0])

    # Simulate the environment using the loaded model
    initial_obs = env.reset()
    simulate_environment(initial_obs, model, 100)
