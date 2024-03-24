
import pickle
from collections import defaultdict
import numpy as np
import logging

def load_tuples(filename):
    with open(filename, 'rb') as f:
        tuples = pickle.load(f)
    return tuples

def compute_metrics(tuples, environment_type):
    _, _, rewards, _, dones, _ = zip(*tuples)
    mean_reward = sum(rewards) / len(rewards)
    
    done_indices = [i for i, done in enumerate(dones) if done]
    episode_lengths = []
    if done_indices:
        episode_lengths.append(done_indices[0] + 1)
        episode_lengths.extend(done_indices[i] - done_indices[i-1] for i in range(1, len(done_indices)))
    
    mean_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
    
    logging.info(f"{environment_type} - Mean Reward: {mean_reward:.4f}, Mean Episode Length: {mean_episode_length:.2f}")
    logging.info(f"{environment_type} - Number of Episodes: {len(episode_lengths)}, Number of steps: {len(tuples)}")

def compute_distributions(tuples, environment_type):
    transitions = defaultdict(lambda: {'next_states': [], 'rewards': [], 'dones': []})
    
    for state, action, reward, next_state, done, _ in tuples:
        transitions[(state, action)]['next_states'].append(next_state)
        transitions[(state, action)]['rewards'].append(reward)
        transitions[(state, action)]['dones'].append(done)
    
    sorted_state_actions = sorted(transitions.keys(), key=lambda x: (x[0], x[1]))

    for state_action in sorted_state_actions:
        data = transitions[state_action]
        logging.info(f"{environment_type} - State-Action: {state_action}")
        for key, values in data.items():
            if key in ['next_states', 'rewards']:
                unique, counts = np.unique(values, return_counts=True)
                distribution = dict(zip(unique, counts))
                logging.info(f"  {environment_type} - {key.capitalize()} Distribution: {distribution}")
            elif key == 'dones':
                true_count = values.count(True)
                false_count = values.count(False)
                logging.info(f"  {environment_type} - Dones Distribution: True - {true_count}, False - {false_count}")

    
if __name__ == "__main__":

    env_name = 'FrozenLake-v1'
    simulator_version = 'v2'
    real_tuples = load_tuples(f"./data/sampled_tuples/sampled_tuples_{env_name}.pkl")
    simulated_tuples = load_tuples(f"./data/simulated_tuples/{env_name}/simulated_tuples_{simulator_version}.pkl")

    logging.basicConfig(filename=f'./logs/environment_analysis_{simulator_version}.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

    # Log the first few tuples for inspection
    logging.info("Sample Real Environment Tuples:")
    for tuple in real_tuples[:5]:  # Adjust the number to change how many tuples are logged
        logging.info(tuple)
    
    logging.info("Sample Simulated Environment Tuples:")
    for tuple in simulated_tuples[:5]:
        logging.info(tuple)

    # Continue with metrics and distributions logging
    logging.info("Real Environment Metrics:")
    compute_metrics(real_tuples, "Real Environment")
    
    logging.info("Simulated Environment Metrics:")
    compute_metrics(simulated_tuples, "Simulated Environment")
    
    logging.info("Real Environment Distributions:")
    compute_distributions(real_tuples, "Real Environment")
    
    logging.info("Simulated Environment Distributions:")
    compute_distributions(simulated_tuples, "Simulated Environment")
