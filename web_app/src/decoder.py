import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32*32*32)
        )

    def forward(self, z):
        out = self.fc(z)
        return out.view(-1, 1, 32, 32, 32)