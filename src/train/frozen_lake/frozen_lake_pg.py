import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PolicyGradientAgent():
    def __init__(self, in_states, h1_nodes, out_actions, learning_rate=0.01):
        self.policy_network = PolicyNetwork(in_states, h1_nodes, out_actions)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        # Convert state to a one-hot-encoded vector
        state = env.reset()[0]

        action_probs = torch.softmax(self.policy_network(state), dim=1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def train(self, log_probs, rewards):
        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

def train_policy_gradient(agent, episodes, render=False, is_slippery=False):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []

        while True:
            action, log_prob = agent.select_action(state_to_dqn_input(state, num_states))
            log_probs.append(log_prob)
            
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            state = new_state

            if done:
                break

        agent.train(log_probs, rewards)
        rewards_per_episode.append(sum(rewards))

    env.close()
    return rewards_per_episode

'''
Converts an state (int) to a tensor representation.
For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

Parameters: state=5, num_states=16
Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
# state = torch.eye(16)[int(state)].long()
def state_to_dqn_input(state:int, num_states:int)->torch.Tensor:
    input_tensor = torch.zeros(num_states)
    input_tensor[state] = 1
    return input_tensor
    
if __name__ == "__main__":
    in_states = 16  # Adjust according to the environment
    h1_nodes = 128  # You can experiment with the number of nodes
    out_actions = 4  # Number of actions in FrozenLake environment

    policy_gradient_agent = PolicyGradientAgent(in_states, h1_nodes, out_actions)
    rewards = train_policy_gradient(policy_gradient_agent, episodes=3000, is_slippery=True)

    # Plotting rewards
    plt.plot(rewards)
    plt.title('Policy Gradient Training on FrozenLake')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human')




agent = PolicyGradientAgent(in_states, h1_nodes, out_actions)

action, log_prob = agent.select_action(state)
