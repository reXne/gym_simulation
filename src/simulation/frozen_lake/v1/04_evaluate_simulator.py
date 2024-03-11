import pickle

def load_tuples(filename):
    with open(filename, 'rb') as f:
        tuples = pickle.load(f)
    return tuples

def compute_metrics(tuples):
    _, _, rewards, _, dones, _ = zip(*tuples)
    mean_reward = sum(rewards) / len(rewards)
    
    done_indices = [i for i, done in enumerate(dones) if done]
    
    # Diagnostic print statements
    print(f"Number of done flags set to True: {sum(dones)}")
    print(f"First 10 simulated tuples: {tuples[:100]}")
    
    # Calculate episode lengths
    episode_lengths = [done_indices[0] + 1]  # Length of the first episode
    for i in range(1, len(done_indices)):
        episode_lengths.append(done_indices[i] - done_indices[i-1])
    
    # Check if episode_lengths is empty
    if not episode_lengths:
        print("Warning: No episodes were completed. Check the 'done' flags in the tuples.")
        mean_episode_length = 0
    else:
        mean_episode_length = sum(episode_lengths) / len(episode_lengths)
    
    return mean_reward, mean_episode_length



if __name__ == "__main__":
    env_name = 'FrozenLake-v1'
    simulator_version = 'v1'
    real_tuples = load_tuples(f"./data/sampled_tuples/sampled_tuples_{env_name}.pkl")
    simulated_tuples = load_tuples(f"./data/simulated_tuples/{env_name}/simulated_tuples_{simulator_version}.pkl")

    real_mean_reward, real_mean_episode_length = compute_metrics(real_tuples)
    simulated_mean_reward, simulated_mean_episode_length = compute_metrics(simulated_tuples)

    print(f"Real Environment - Mean Reward: {real_mean_reward}, Mean Episode Length: {real_mean_episode_length}")
    print(f"Simulated Environment - Mean Reward: {simulated_mean_reward}, Mean Episode Length: {simulated_mean_episode_length}")
