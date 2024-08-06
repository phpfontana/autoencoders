import torch.nn as nn
from torch import Tensor
from ..base import BaseModel

class AE(BaseModel):
    def __init__(self, latent_dim, hidden_dim):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat