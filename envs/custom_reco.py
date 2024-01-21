import gym
from gym import spaces
import numpy as np

class CustomerBehaviorEnv(gym.Env):
    def __init__(self, initial_state_mean, initial_state_std):
        super(CustomerBehaviorEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        # Initialize customer characteristics
        self.initial_state_mean = initial_state_mean
        self.initial_state_std = initial_state_std

        # Set maximum number of steps (episodes)
        self.max_steps = 1000
        self.current_step = 0

    def reset(self):
        # Sample initial state from the provided distribution
        initial_customer_tenure = np.random.randint(1, 11)  # Assuming a range for customer_tenure
        initial_time_since_action = np.random.randint(0, 11)  # Assuming a range for time_since_action
        
        # Sample initial number of people in home (categorical)
        initial_num_people_in_home = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])

        # Sample initial number of lines (categorical)
        initial_num_lines = np.random.choice([1, 2, 3], p=[0.8, 0.1, 0.1])

        # Sample initial service usage (integer)
        initial_service_usage = np.random.randint(0, 5)
        
        initial_state = np.array([initial_customer_tenure, initial_time_since_action,
                                initial_num_people_in_home, initial_num_lines, initial_service_usage], dtype=np.float32)

        # Clip to ensure non-negative values
        initial_state = np.clip(initial_state, 0, np.inf)

        # Update customer characteristics
        (self.customer_tenure, self.time_since_action, self.num_people_in_home,
        self.num_lines, self.service_usage) = initial_state

        self.current_step = 0

        # Return initial state
        return initial_state


    def multinomial_logit(self, features):
        # Coefficients for the multinomial logit model
        coef_tenure = 0.00001
        coef_time = 0.005
        coef_people = 0.005
        coef_lines = -0.05
        coef_usage = 0.05
        coef_action = 0.1

        # Calculate the logits for each category
        logits = (
            coef_tenure * self.customer_tenure +
            coef_time * self.time_since_action +
            coef_people * self.num_people_in_home +
            coef_lines * self.num_lines +
            coef_usage * self.service_usage + 
            coef_action * features[action]
        )

        # Sigmoid function to get the probability of a positive response
        probability_positive_response = 1 / (1 + np.exp(-logits))

        return probability_positive_response
    
    def step(self, action):
        # Model customer behavior based on characteristics
        features = np.array([self.customer_tenure, self.time_since_action, self.num_people_in_home,
                             self.num_lines, self.service_usage, action], dtype=np.float32)

        # Probability of positive response based on multinomial logit model
        probability_positive_response = self.multinomial_logit(features) * 0.15
        # print(f"probability positive response {probability_positive_response}")

        customer_response = 1 if np.random.rand() < (probability_positive_response ) else 0

        if customer_response == 1:
             self.num_lines += 1
        # Update customer characteristics based on the action
        if action == 0:
            # Action is 0: Customer was not contacted
            self.time_since_action += 1
            
            reward = self.num_lines 
            
        if action == 1:  
            # Action is 1: Customer was not contacted
            self.time_since_action = 0
        
            reward = self.num_lines - 0.05  
    
        # Update other customer characteristics
        self.service_usage += np.random.randint(0, 5)
        self.customer_tenure += 1

        # Construct the next state
        next_state = np.array([self.customer_tenure, self.time_since_action, self.num_people_in_home,
                             self.num_lines, self.service_usage], dtype=np.float32)

        # Done flag is set randomly for simplicity
        done = np.random.rand() < 0.1

        # Increment the step counter
        self.current_step += 1

        # Check if the maximum number of steps is reached
        done = done or self.current_step >= self.max_steps

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Optional: Implement visualization of the environment
        pass

    def close(self):
        # Optional: Implement any necessary cleanup
        pass

# Example usage with initial state distribution mean=10, std=2
initial_state_mean = 10
initial_state_std = 2

env = CustomerBehaviorEnv(initial_state_mean, initial_state_std)

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
