import gym
from gym import spaces
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Discrete(5)  # 5 discrete states

        # Set maximum number of steps
        self.max_steps = 20
        self.current_step = 0

        # Define hardcoded transition matrix
        self.transition_matrix = np.array([
            # Transition probabilities for action 0
            [
                [0.3, 0.2, 0.1, 0.2, 0.2],
                [0.1, 0.4, 0.1, 0.2, 0.2],
                [0.2, 0.1, 0.4, 0.1, 0.2],
                [0.2, 0.2, 0.1, 0.3, 0.2],
                [0.2, 0.2, 0.2, 0.1, 0.3]
            ],
            # Transition probabilities for action 1
            [
                [0.4, 0.3, 0.1, 0.1, 0.1],
                [0.1, 0.4, 0.2, 0.1, 0.2],
                [0.2, 0.1, 0.4, 0.1, 0.2],
                [0.1, 0.2, 0.1, 0.3, 0.3],
                [0.2, 0.1, 0.2, 0.2, 0.3]
            ]
        ])

        # Define reward probabilities for each state-action-next_state combination
        self.reward_probabilities = np.array([
            # Reward probabilities for action 0
            [
                [
                    [0.8, 0.2],
                    [0.7, 0.3],
                    [0.9, 0.1],
                    [0.6, 0.4],
                    [0.5, 0.5]
                ],
                [
                    [0.3, 0.7],
                    [0.2, 0.8],
                    [0.1, 0.9],
                    [0.4, 0.6],
                    [0.6, 0.4]
                ],
                [
                    [0.2, 0.8],
                    [0.1, 0.9],
                    [0.3, 0.7],
                    [0.5, 0.5],
                    [0.4, 0.6]
                ],
                [
                    [0.5, 0.5],
                    [0.6, 0.4],
                    [0.4, 0.6],
                    [0.2, 0.8],
                    [0.1, 0.9]
                ],
                [
                    [0.4, 0.6],
                    [0.3, 0.7],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.8, 0.2]
                ]
            ],
            # Reward probabilities for action 1
            [
                [
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.1, 0.9],
                    [0.4, 0.6],
                    [0.5, 0.5]
                ],
                [
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.9, 0.1],
                    [0.6, 0.4],
                    [0.4, 0.6]
                ],
                [
                    [0.8, 0.2],
                    [0.9, 0.1],
                    [0.7, 0.3],
                    [0.5, 0.5],
                    [0.6, 0.4]
                ],
                [
                    [0.5, 0.5],
                    [0.4, 0.6],
                    [0.6, 0.4],
                    [0.8, 0.2],
                    [0.9, 0.1]
                ],
                [
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.3, 0.7],
                    [0.2, 0.8]
                ]
            ]
        ])

        # Define done probabilities for each state-action-next_state combination
        self.done_probabilities = np.array([
            # Done probabilities for action 0
            [
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.7, 0.3],
                    [0.6, 0.4],
                    [0.5, 0.5]
                ],
                [
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.1, 0.9],
                    [0.4, 0.6],
                    [0.6, 0.4]
                ],
                [
                    [0.5, 0.5],
                    [0.6, 0.4],
                    [0.4, 0.6],
                    [0.7, 0.3],
                    [0.8, 0.2]
                ],
                [
                    [0.3, 0.7],
                    [0.2, 0.8],
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.2, 0.8]
                ],
                [
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.5, 0.5],
                    [0.4, 0.6]
                ]
            ],
            # Done probabilities for action 1
            [
                [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.4, 0.6],
                    [0.5, 0.5]
                ],
                [
                    [0.8, 0.2],
                    [0.7, 0.3],
                    [0.9, 0.1],
                    [0.6, 0.4],
                    [0.4, 0.6]
                ],
                [
                    [0.5, 0.5],
                    [0.4, 0.6],
                    [0.6, 0.4],
                    [0.3, 0.7],
                    [0.2, 0.8]
                ],
                [
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.2, 0.8]
                ],
                [
                    [0.4, 0.6],
                    [0.5, 0.5],
                    [0.8, 0.2],
                    [0.6, 0.4],
                    [0.7, 0.3]
                ]
            ]
        ])

        # Specify terminal states
        self.terminal_states = [2, 4]

    def reset(self):
        # Reset to initial state
        self.current_state = 0
        self.current_step = 0

        return self.current_state

    def step(self, action):
        # Check if action is valid
        assert self.action_space.contains(action), "Invalid action"

        # Transition to a new state based on the action
        prob_distribution = self.transition_matrix[action][self.current_state]
        new_state = np.random.choice(self.observation_space.n, p=prob_distribution)

        # Sample reward from the reward probabilities matrix
        reward = np.random.choice([0, 1], p=self.reward_probabilities[action][self.current_state][new_state])

        # Sample done flag from the done probabilities matrix
        done = np.random.choice([0, 1], p=self.done_probabilities[action][self.current_state][new_state])

        # Update current state
        self.current_state = new_state

        # Increment the step counter
        self.current_step += 1

        # Check if the maximum number of steps is reached or if the current state is terminal
        done = done or self.current_step >= self.max_steps or self.current_state in self.terminal_states

        return new_state, reward, done, {}

    def render(self, mode='human'):
        # Optional: Implement visualization of the environment
        pass

    def close(self):
        # Optional: Implement any necessary cleanup
        pass

# Example usage
env = SimpleEnv()

# Reset the environment to get the initial state
initial_state = env.reset()

# Take a random action
action = env.action_space.sample()

# Perform a step in the environment
next_state, reward, done, _ = env.step(action)

# Print the results
print("Initial State:", initial_state)
print("Action:", action)
print("Next State:", next_state)
print("Reward:", reward)
print("Done:", done)
