import torch
import numpy as np
from src.simulation.simulator import RSSM  # Assuming RSSM is the model class

class SimulatedGymEnvironment:
    def __init__(self, model_path, state_dim, action_dim, hidden_dim=8, latent_dim=2):
        self.model = RSSM(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode
        self.current_state = None

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        # Assuming the initial state is either known or randomly chosen, for example:
        self.current_state = torch.zeros(state_dim)  # This should be replaced with a proper initial state
        return self.current_state.numpy()

    def step(self, action):
        """
        Steps the environment according to the action and returns the next state, reward, done, and info.
        """
        # Convert action to tensor and possibly one-hot encode if needed
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # Predict the next state and reward
        with torch.no_grad():  # No need to track gradients
            next_state, reward, _, _ = self.model.predict_next_state(self.current_state.unsqueeze(0), action_tensor)
        
        done = self.check_done(next_state)  # Implement this method based on the environment's termination conditions
        self.current_state = next_state.squeeze(0)  # Remove batch dimension
        
        # Convert tensors to numpy arrays for compatibility with Gym
        next_state_np = next_state.squeeze(0).numpy()
        reward_np = reward.item()
        
        info = {}  # Additional info, if any (usually empty in simulated environments)
        return next_state_np, reward_np, done, info

    def check_done(self, next_state):
        """
        Check if the episode should end based on the next state.
        This method needs to be customized based on the specific termination conditions of your environment.
        """
        # Example condition, this should be replaced with real conditions
        done = torch.rand(1).item() < 0.1  # Randomly end an episode with a 10% chance
        return done

# Example usage
if __name__ == "__main__":
    env_name = 'CartPole-v1'
    model_path = f'./data/models/rssm_model_{env_name}.pth'
    if env_name == 'CartPole-v1':
        state_dim = 4
        action_dim = 2  # Assuming action_dim is correctly defined for your model
    elif env_name == 'FrozenLake-v1':
        state_dim = 16
        action_dim = 4

    simulated_env = SimulatedGymEnvironment(model_path, state_dim, action_dim)
    state = simulated_env.reset()
    done = False
    while not done:
        action = np.random.choice(action_dim)  # Random action, replace with your policy
        next_state, reward, done, info = simulated_env.step(action)
        print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        state = next_state
