import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sdf_dataset import SDFDataset
from autoencoder_model import Encoder, Decoder
import os
import matplotlib.pyplot as plt

train_losses = []


# Load dataset
# Change the path to your dataset directory here
dataset = SDFDataset('../data/sdf')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Models
latent_size = 128
encoder = Encoder(latent_size)
decoder = Decoder(latent_size)

# Optimizer
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(70):
    train_loss = 0
    for sdf, _ in dataloader:
        z = encoder(sdf)
        pred = decoder(z)
        loss = loss_fn(pred, sdf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calculate the cumulative losses in this epoch
    train_loss = train_loss / len(dataloader)
    
    # Added cumulative losses to lists for later display
    train_losses.append(train_loss)

    print(f"Epoch {epoch+1} Loss: {train_loss/len(dataloader):.4f}")

# Save models
os.makedirs("checkpoints", exist_ok=True)
torch.save(encoder.state_dict(), "checkpoints/encoder.pth")
torch.save(decoder.state_dict(), "checkpoints/decoder.pth")
print("✅ Autoencoder trained and saved!")

# Plot training loss
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()