import gym
import random
import pickle
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/01_sampling_tuples.log',  # Name of the log file
                    filemode='w')  # Append mode; use 'w' for overwrite mode

def save_tuples(tuples, filename="./data/sampled_tuples/sampled_tuples.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    logging.info(f"Tuples saved to {filename}")

def simulate_environment():
    # Initialize environment
    env = gym.make('CartPole-v1')
    num_episodes = 10000
    tuples = []

    total_rewards = 0
    total_dones = 0
    reward_distribution = {0: 0, 1: 0}  # To keep track of reward counts

    logging.info('observation_space')
    logging.info(env.observation_space)
    logging.info('action_space')
    logging.info(env.action_space)
    
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        if isinstance(observation, tuple):  # Check if the observation is a tuple
            observation, _ = observation  # Extract only the array part of the observation
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            old_observation = observation
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
            tuples.append((old_observation, action, reward, observation, done))
            
            # Update reward distribution
            reward_distribution[reward] += 1
            
        total_rewards += episode_reward
        if done:
            total_dones += 1

    env.close()
    
    avg_reward = total_rewards / num_episodes

    stats = {
        "average_reward": avg_reward,
        "total_episodes": num_episodes,
        "total_dones": total_dones,
        "reward_distribution": reward_distribution
    }
    
    return tuples, stats

def main():
    tuples, stats = simulate_environment()
    save_tuples(tuples)
    
    logging.info("Summary Statistics:")
    logging.info(f"Average Reward: {stats['average_reward']}")
    logging.info(f"Total Episodes: {stats['total_episodes']}")
    logging.info(f"Total Dones: {stats['total_dones']}")
    logging.info(f"Reward Distribution: {stats['reward_distribution']}")

if __name__ == "__main__":
    main()
