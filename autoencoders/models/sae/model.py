import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List
from ..base import BaseModel

class SparseAE(BaseModel):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], # (batch_size, channels, height, width)
                 hidden_dim: List[int], 
                 latent_dim: int, 
                 sparsity: float): 
        super(SparseAE, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sparsity = sparsity

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

    def loss_function(self, x: Tensor, x_hat: Tensor) -> Tensor:
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        sparsity_loss = torch.mean(self.sparsity * torch.log(self.sparsity / self.encoder(x).mean(dim=0)) + (1 - self.sparsity) * torch.log((1 - self.sparsity) / (1 - self.encoder(x).mean(dim=0))))
        return reconstruction_loss + sparsity_loss
