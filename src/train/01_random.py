# pip install gym==0.18.0
# print( gym.__version__ )

import gym
import torch

def select_action(env, strategy, state=None, model=None):
    """
    Select action based on strategy.
    :param env: Gym environment.
    :param strategy: 'random', 'heuristic', or 'deterministic'.
    :param state: Current state of the environment.
    :param model: Trained model (used only for deterministic strategy).
    :return: Chosen action.
    """
    if strategy == 'random':
        return env.action_space.sample()
    elif strategy == 'heuristic':
        # Assuming the 3rd value in the state is the pole_angle
        pole_angle = state[2]
        if pole_angle > 0:
            return 1  # Move to the right
        else:
            return 0  # Move to the left
    elif strategy == 'deterministic':
        if model:
            with torch.no_grad():
                return model.select_action(torch.FloatTensor(state), deterministic=True)
        else:
            raise ValueError("Model is required for deterministic strategy")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_policy(env, strategy='random', model=None, render=True, num_episodes=100):
    score = 0
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        done = False
        ep_r = 0

        while not done:
            a = select_action(env, strategy, state, model)
            state_prime, r, done, _, _ = env.step(a)
            ep_r += r
            state = state_prime
            if render:
                env.render()
        score += ep_r
        print(f'{strategy.capitalize()} Policy: Episode {episode}, Episode reward {ep_r}, Running average {score/(episode+1)}')

    print(f'Score / num_episodes {score/num_episodes}')
    env.close()

def main():
    env = gym.make('CartPole-v1', render_mode='human')
    print('observation_space:\n', env.observation_space)
    print('\naction_space:\n', env.action_space)
    num_episodes = 100

    # Evaluate Random Policy
    #evaluate_policy(env=env, strategy='random', num_episodes=num_episodes)

    # Evaluate Heuristic Policy
    evaluate_policy(env=env, strategy='heuristic', num_episodes=num_episodes)

    # For deterministic, you would need a model.
    # e.g., evaluate_policy(env=env, strategy='deterministic', model=trained_model, num_episodes=num_episodes)

if __name__ == "__main__":
    main()
