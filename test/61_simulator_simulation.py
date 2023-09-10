import numpy as np
import torch
import pickle
from simulator2 import VAE, VAESimulator
import hydra
from omegaconf import DictConfig
import pandas as pd
import logging
import time

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, 
                filename=cfg.logs.files.vae_simulation,
                filemode='w',
                format='%(asctime)s - %(levelname)s - \n%(message)s')
    
    tuples_train_file = cfg.data.files.tuples_train_resampled_file
    #tuples_train = pd.read_pickle(tuples_train_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 54
    action_dim = 1
    hidden_dim = 64
    latent_dim = 32
    reward_dim = 1
    done_dim = 1
    input_dim = state_dim + action_dim
    output_dim = state_dim + reward_dim + done_dim 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model and load the trained parameters
    model = VAE(input_dim, hidden_dim, latent_dim, output_dim).to(device)
    model.load_state_dict(torch.load('./models/vae_model.pth'))
 
    simulator = VAESimulator(model, state_dim, action_dim, reward_dim, done_dim, latent_dim, device)

    # Load your initial states and actions
    with open(tuples_train_file, "rb") as f:
        data = pickle.load(f)
    S, A, _, _, _, _ = data  # Assuming that you don't need the rest of the tuple data here

    # Generate and save tuples
    S_gen = []
    A_gen = []
    R_gen = []
    Sp_gen = []
    D_gen = []

    for state, action in zip(S, A.reshape(-1, 1)):
        next_state, next_reward, next_done = simulator.generate_next_state_reward_done(state, action)
        logging.info(f"State shape: {state.shape}, action: {action.shape}")
        logging.info(f"next_state: {next_state.shape}, next_reward: {next_reward.shape}, next_done: ")
        logging.info(f"next_state: {next_state}, next_reward: {next_reward}, next_done: {next_done}")
        S_gen.append(state)
        A_gen.append(action)
        R_gen.append(next_reward)  # Unpack the reward value from the numpy array
        Sp_gen.append(next_state)
        D_gen.append(int(next_done))
        
    S_gen = np.array(S_gen)
    A_gen = np.array(A_gen)
    R_gen = np.array(R_gen)
    Sp_gen = np.array(Sp_gen)
    D_gen = np.array(D_gen)

    tuples = [S_gen, A_gen, R_gen, Sp_gen, D_gen]

    with open('./data/40_simulator/simulated_tuples.pkl', 'wb') as f:
        pickle.dump(tuples, f)

    end_time = time.time()
    elapsed_time = end_time - start_time       
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
