# Assuming 'initial_states' is a DataLoader containing your initial state data

from src.simulation.frozen_lake.v3.simulator import VAEInitialStateModel

model = VAEInitialStateModel(state_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def generate_random_states(model, num_samples, latent_dim):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Sample from the standard Gaussian distribution
        z = torch.randn(num_samples, latent_dim)
        # Generate data by decoding the sampled latent variables
        generated_data = model.decoder(z).cpu()
    return generated_data


for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(initial_states):
        data = data.to(device)  # Assuming you're using a device like 'cuda' or 'cpu'
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss
        loss = vae_loss(recon_batch, data, mu, logvar)
        train_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
    # Logging
    print(f'Epoch {epoch}, Loss: {train_loss / len(initial_states.dataset)}')
