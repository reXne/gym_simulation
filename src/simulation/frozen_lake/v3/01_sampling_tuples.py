import gymnasium as gym
import os
import pickle
import logging
import numpy as np
from collections import defaultdict

def save_tuples(tuples, env_name, filename="./data/sampled_tuples/sampled_tuples"):
    """Save the collected tuples to a file."""
    filename = f"{filename}_{env_name}.pkl"  # Append env_name to the filename
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    logging.info(f"Tuples saved to {filename}")

def sample_environment(env_name, map_name="4x4", is_slippery=True, render=False):
    """Simulate the environment and collect tuples of observations, actions, rewards, and new observations."""
    env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, render_mode='human' if render else None)

    num_episodes = 10000
    tuples = []
    transitions = defaultdict(lambda: {'next_states': [], 'rewards': [], 'dones': []})

    logging.info(f'Environment: {env_name}')
    logging.info(f'Observation Space: {env.observation_space}')
    logging.info(f'Action Space: {env.action_space}')
    
    for _ in range(num_episodes):
        observation = env.reset()
        is_initial_state = True 
        if isinstance(observation, tuple):
          observation = observation[0]
        
        done = False
        while not done:
            action = env.action_space.sample()
            old_observation = observation
            observation, reward, done, _, _ = env.step(action)
            if isinstance(observation, tuple):
                 observation = observation[0]
            tuples.append((old_observation, action, reward, observation, done, is_initial_state))
            is_initial_state = False
            
            transitions[(old_observation, action)]['next_states'].append(observation)
            transitions[(old_observation, action)]['rewards'].append(reward)
            transitions[(old_observation, action)]['dones'].append(done)
    
    env.close()
    
     # Extract actions and rewards from tuples for statistics
    actions = [t[1] for t in tuples]
    rewards = [t[2] for t in tuples]

    # Calculate statistics
    action_distribution = {action: actions.count(action) for action in set(actions)}
    reward_distribution = {reward: rewards.count(reward) for reward in set(rewards)}
    mean_reward = np.mean(rewards)
    total_dones = sum([1 for t in tuples if t[4]])

    # Log statistics
    logging.info(f"Total Episodes: {num_episodes}")
    logging.info(f"Total Dones: {total_dones}")
    logging.info(f"Mean Reward: {mean_reward:.2f}")
    logging.info(f"Action Distribution: {action_distribution}")
    logging.info(f"Reward Distribution: {reward_distribution}")
    
    """Log distributions of rewards, next states, and dones given state-action combinations."""
    for state_action, data in transitions.items():
        logging.info(f"State-Action: {state_action}")
        for key, values in data.items():
            if key in ['next_states', 'rewards']:
                unique, counts = np.unique(values, return_counts=True)
                distribution = dict(zip(unique, counts))
                logging.info(f"  {key.capitalize()} Distribution: {distribution}")
            elif key == 'dones':
                true_count = values.count(True)
                false_count = values.count(False)
                logging.info(f"  Dones Distribution: True - {true_count}, False - {false_count}")
                
    return tuples




def main():
    env_name = 'FrozenLake-v1'
    
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f'./logs/01_sampling_tuples_{env_name}.log',  # Adjusted the log file name
                        filemode='w')  # Overwrite mode

    is_slippery = True
    render = False

    tuples = sample_environment(env_name=env_name, is_slippery=is_slippery, render=render)
    save_tuples(tuples, env_name)

    

if __name__ == "__main__":
    main()

