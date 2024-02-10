import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from src.vae.model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 2
BATCH_SIZE = 32
LR_RATE = 3e-4

# Dataset Loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, _) in loop:
            images = images.to(device).view(images.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(images)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, images)
            kl_div = torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def inference(model, dataset, digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, INPUT_DIM))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"./data/vae/generated_{digit}_ex{example}.png")

# Training
train(model, train_loader, optimizer, loss_fn, DEVICE)

# Inference and Generation
for idx in range(10):
    inference(model, dataset, idx, num_examples=5)
    
    
    
    
import matplotlib.pyplot as plt

# Visualize a few samples
sample_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
for images, labels in sample_loader:
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()
    
# Directory to save generated images
os.makedirs("./data/vae/generated", exist_ok=True)

# Set the model to evaluation mode
model.eval()

# Number of examples to generate for each digit
num_examples = 5

# Iterate over digits
for digit in range(10):
    # Extract an example image for each digit
    real_images = [x for x, y in dataset if y == digit][:num_examples]

    encodings_digit = []
    for real_image in real_images:
        with torch.no_grad():
            mu, sigma = model.encode(real_image.view(1, INPUT_DIM))
        encodings_digit.append((mu, sigma))

    # Create a subplot for each real and generated pair
    plt.figure(figsize=(12, 5))
    
    for i in range(num_examples):
        # Plot real image
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(real_images[i].squeeze().numpy(), cmap='gray')
        plt.title(f"Label: {digit} (Real)")
        plt.axis('off')

        # Plot generated image
        mu, sigma = encodings_digit[i]
        for _ in range(5):  # Generate 5 samples for each example
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = model.decode(z)
            out = out.view(-1, 1, 28, 28)

            plt.subplot(2, num_examples, num_examples + i * 5 + _ + 1)
            plt.imshow(out.squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Digit: {digit} (Generated)")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"./data/vae/generated/generated_{digit}.png")
    plt.show()