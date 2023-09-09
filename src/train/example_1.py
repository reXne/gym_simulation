import pandas as pd
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def evaluate_policy(env, render=False, num_episodes = 100):

    score = 0
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        ep_r = 0
        steps = 0

        while not done:
            a = env.action_space.sample()  # Random action
            # a = model.select_action(s, deterministic=True)
            s_prime, r, done, _, _ = env.step(a)
            ep_r+= r
            steps += 1
            s = s_prime
            if render:
                env.render()
        score += ep_r
        print(f'Random Policy: Episode {episode}, Episode reward {ep_r}, Running average {score/(episode+1)}')

    print(f'Score / num_episodes {score/num_episodes}')

    env.close()

def main():
    env = gym.make('CartPole-v1', render_mode='human')

    print('observation_space:\n')
    print(env.observation_space)
    print()
    print('action_space:\n')
    print(env.action_space)

    num_episodes = 100

    evaluate_policy(env=env, num_episodes=num_episodes)

if __name__ == "__main__":
    main()


