import os
import torch
import numpy as np
from autoencoder_model import Encoder
from sdf_dataset import SDFDataset
import glob

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load encoder
encoder = Encoder(latent_size=128).to(device)
encoder.load_state_dict(torch.load("../checkpoints/encoder.pth", map_location=device))
encoder.eval()

# Change the path to your dataset directory here
CATEGORY_DIR = "../data/sdf"
OUTPUT_LATENTS = "../latents"

os.makedirs(OUTPUT_LATENTS, exist_ok=True)

def load_voxel(path):
    data = np.load(path)['sdf']  # or whatever your key is
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64, 64]
    return data.to(device)

for category in os.listdir(CATEGORY_DIR):
    latent_sum = torch.zeros((128,), device=device)
    count = 0
    for file in glob.glob(os.path.join(CATEGORY_DIR, category, "*.npz")):
        voxel = load_voxel(file)
        with torch.no_grad():
            latent = encoder(voxel).squeeze()
        latent_sum += latent
        count += 1
    avg_latent = latent_sum / count
    torch.save(avg_latent, f"{OUTPUT_LATENTS}/{category}.pt")
    print(f"✅ Saved average latent for {category}")