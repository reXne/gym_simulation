import gym
import random
import pickle

def simulate_environment():
    # Initialize environment
    env = gym.make('CartPole-v1')
    num_episodes = 1000
    tuples = []

    for _ in range(num_episodes):
        observation = env.reset()
        if isinstance(observation, tuple):  # Check if the observation is a tuple
            observation, _ = observation  # Extract only the array part of the observation
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            old_observation = observation
            observation, reward, done, truncated, info = env.step(action)
            tuples.append((old_observation, action, reward, observation, done))

    env.close()
    
    return tuples

def save_tuples(tuples, filename="./data/sampled_tuples/tuples_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(tuples, f)
    print(f"Tuples saved to {filename}")

def main():
    tuples = simulate_environment()
    save_tuples(tuples)

if __name__ == "__main__":
    main()
