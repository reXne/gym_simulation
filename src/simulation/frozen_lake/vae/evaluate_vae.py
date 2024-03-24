import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.simulation.frozen_lake.vae.train_vae import VAEInitialStateModel, to_one_hot
from torch.distributions import (
    Normal, Bernoulli, Beta, Binomial, Categorical, 
    OneHotCategorical, Independent
)
def load_trained_vae_model(model_path, state_dim, hidden_dim, latent_dim):
    model = VAEInitialStateModel(state_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_samples(model, num_samples):
    with torch.no_grad():
        states_logits = model.sample(num_samples)
        states_dist = Categorical(logits=states_logits)
        states_idx = states_dist.sample()
        states_sampled = torch.nn.functional.one_hot(states_idx, num_classes=states_logits.shape[-1])
    return states_sampled

def load_original_data(filepath):
    with open(filepath, 'rb') as f:
        tuples = pickle.load(f)
    states, _, _, _, _, _ = zip(*tuples)
    original_data = torch.stack([to_one_hot(s, 16) for s in states])
    return original_data

def visualize_with_pca(original_data, generated_data, save_path='PCA_Visualization.png'):
    pca = PCA(n_components=2)
    combined_data = torch.cat((original_data, generated_data), 0)
    reduced_data = pca.fit_transform(combined_data.detach().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(original_data), 0], reduced_data[:len(original_data), 1], color='red', label='Original')
    plt.scatter(reduced_data[len(original_data):, 0], reduced_data[len(original_data):, 1], color='blue', label='Generated')
    plt.legend()
    plt.title('PCA Visualization')
    plt.savefig(save_path)
    plt.close()

def visualize_with_tsne(original_data, generated_data, save_path='tSNE_Visualization.png'):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    combined_data = torch.cat((original_data, generated_data), 0)
    reduced_data = tsne.fit_transform(combined_data.detach().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(original_data), 0], reduced_data[:len(original_data), 1], color='red', label='Original')
    plt.scatter(reduced_data[len(original_data):, 0], reduced_data[len(original_data):, 1], color='blue', label='Generated')
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig(save_path)
    plt.close()

def main():
    # Configuration
    model_path = './data/models/vae_initial_state_model_vae.pth'
    data_path = './data/sampled_tuples/sampled_tuples_FrozenLake-v1.pkl'
    state_dim = 16
    hidden_dim = 8
    latent_dim = 4
    num_samples = 1000  # Adjust based on your needs

    # Load the model
    model = load_trained_vae_model(model_path, state_dim, hidden_dim, latent_dim)

    # Generate samples
    generated_samples = generate_samples(model, num_samples)

    # Load original data
    original_data = load_original_data(data_path)

    # Visualize with PCA
    visualize_with_pca(original_data, generated_samples, save_path='data/plots/PCA_Visualization.png')

    # Visualize and save t-SNE plot
    visualize_with_tsne(original_data, generated_samples, save_path='data/plots/tSNE_Visualization.png')

if __name__ == "__main__":
    main()
