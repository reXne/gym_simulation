#https://stackoverflow.com/questions/70191012/pytorch-ppo-implementation-for-cartpole-v0-getting-stuck-in-local-optima

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt

def running_average(x, n):
    N = n
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N] # matrix multiplication operator: np.mul
        y[i] /= N
    return y

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, learning_rate=0.0003, epsilon_clipping=0.3, update_epochs=10):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1)
        ).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epsilon_clipping = epsilon_clipping
        self.update_epochs = update_epochs

    def forward(self, X):
        return self.model(X)

    
    def predict(self, state):
        if state.ndim < 2:
            action_probs = self.model(torch.FloatTensor(state).unsqueeze(0).float())
        else: 
            action_probs = self.model(torch.FloatTensor(state))

        return action_probs.squeeze(0).data.numpy()

   
    def update(self, states, actions, deltas, old_prob):
  
        batch_size = len(states)
        state_batch = torch.Tensor(states)
        action_batch = torch.Tensor(actions)
        delta_batch = torch.Tensor(deltas)
        old_prob_batch = torch.Tensor(old_prob)
        for k in range(self.update_epochs):
            pred_batch = self.model(state_batch)

            prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

            ratio = torch.exp(torch.log(prob_batch) - torch.log(old_prob_batch))

            clipped = torch.clamp(ratio, 1 - self.epsilon_clipping, 1 + self.epsilon_clipping) * delta_batch
            loss_r = -torch.min(ratio*delta_batch, clipped)
            loss = torch.mean(loss_r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




class CriticNetwork(nn.Module):
    def __init__(self, state_dim, learning_rate=0.001):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


    def forward(self, X):
        return self.model(X)

    def predict(self, state):
        if state.ndim < 2:
            values = self.model(torch.FloatTensor(state).unsqueeze(0).float())
        else:
            values = self.model(torch.FloatTensor(state))

        return values.data.numpy()

  
    def update(self, states, targets):
        
        state_batch = torch.Tensor(states)
        target_batch = torch.Tensor(targets)
        pred_batch = self.model(state_batch)
        loss = torch.nn.functional.mse_loss(pred_batch, target_batch.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    


def train_ppo_agent(env, episode_length, max_episodes, gamma, visualize_step, learning_rate_actor=0.0003, learning_rate_critic=0.001, epsilon_clipping=0.2, actor_update_epochs=10):

  
    model_actor = ActorNetwork(env.observation_space.shape[0], env.action_space.n, learning_rate=learning_rate_actor,
                               epsilon_clipping=epsilon_clipping, update_epochs=actor_update_epochs)
    model_critic = CriticNetwork(env.observation_space.shape[0], learning_rate=learning_rate_critic)



    EPISODE_LENGTH = episode_length
    MAX_EPISODES = max_episodes
    GAMMA = gamma
    VISUALIZE_STEP = max(1, visualize_step)
    score = []


    for episode in range(MAX_EPISODES):
        curr_state = env.reset()
        done = False
        all_episode_t = []
        score_episode = 0
        for t in range(EPISODE_LENGTH):
            act_prob = model_actor.predict(curr_state)
            action = np.random.choice(np.array(list(range(env.action_space.n))), p=act_prob)
            value = model_critic.predict(curr_state)
            prev_state = curr_state
            curr_state, reward, done, info = env.step(action)
            score_episode += reward
            e_t = {'state': prev_state, 'action':action, 'action_prob':act_prob[action],'reward': reward, 'value': value}
            all_episode_t.append(e_t)
            if done:
                break
        score.append(score_episode)

        episode_values = [all_episode_t[t]['value'] for t in range(len(all_episode_t))]
        next_state_estimates = [episode_values[i].item() for i in range(1, len(episode_values))]
        next_state_estimates.append(0)
        boostrap_estimate = []
        for t in range(len(all_episode_t)):
            G = all_episode_t[t]['reward'] + GAMMA * next_state_estimates[t]
            boostrap_estimate.append(G)

        episode_target = np.array(boostrap_estimate)
        episode_values = np.array(episode_values)
        # compute the advantage for each state in the episode: R_{t+1} + \gamma * V(S_{t+1}) - V_{t}
        adv_batch = episode_target-episode_values
       
        state_batch = np.array([all_episode_t[t]['state'] for t in range(len(all_episode_t))])
        action_batch = np.array([all_episode_t[t]['action'] for t in range(len(all_episode_t))])
        old_actor_prob = np.array([all_episode_t[t]['action_prob'] for t in range(len(all_episode_t))])
       
        model_actor.update(state_batch, action_batch, adv_batch, old_actor_prob)
       
        model_critic.update(state_batch, episode_target)

        # print the status after every VISUALIZE_STEP episodes
        if episode % VISUALIZE_STEP == 0 and episode > 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score[-VISUALIZE_STEP:-1])))
            # domain knowledge applied to stop training: if the average score across last 100 episodes is greater than 195, game is solved
            if np.mean(score[-100:-1]) > 195:
                break


    # Training plot: Episodic reward over Training Episodes
    score = np.array(score)
    avg_score = running_average(score, visualize_step)
    plt.figure(figsize=(15, 7))
    plt.ylabel("Episodic Reward", fontsize=12)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.plot(score, color='gray', linewidth=1)
    plt.plot(avg_score, color='blue', linewidth=3)
    plt.scatter(np.arange(score.shape[0]), score, color='green', linewidth=0.3)
    plt.savefig("temp/cartpole_ppo_training_plot.pdf")

    # return the trained models
    return model_actor, model_critic

def main():
    env = gym.make('CartPole-v0')
    episode_length = 300
    n_episodes = 5000
    gamma = 0.99
    vis_steps = 100
    learning_rate_actor = 0.0003
    actor_update_epochs = 10
    epsilon_clipping = 0.2
    learning_rate_critic = 0.001
   
    # train the PPO agent
    model_actor, model_critic = train_ppo_agent(env, episode_length, n_episodes, gamma, vis_steps,
                                               learning_rate_actor=learning_rate_actor,
                                               learning_rate_critic=learning_rate_critic,
                                               epsilon_clipping=epsilon_clipping,
                                               actor_update_epochs=actor_update_epochs)