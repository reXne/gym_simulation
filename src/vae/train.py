import torch
import torchvision.datasets as datasets # Standard datasets
from tqdm import tqdm
from torch import nn, optim
from vae.model import VariationalAutoEncoder
from torchvision import transforms # ensures each pixel is between 0 and 1
from torchvision.utils import save_image
from torch.utils.data import DataLoader # Gives easier dataset managment by creating mini batches etc.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4 # karpathy constant
 
# Dataset Loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum") 

# Start Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # Forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)
        
        # Coompute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = torch.sum(1 + torch.log(sigma.pow(2))- mu.pow(2) - sigma.pow(2))
        
        # Backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())


def inference(digit, num_examples=1):
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
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))
        
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"./data/vae/generated_{digit}_ex{example}.png")
        
for idx in range(10):
    inference(idx, num_examples=5)

