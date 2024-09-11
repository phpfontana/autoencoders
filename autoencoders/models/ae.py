import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass


class AE(nn.Module):
    """Autoencoder (AE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dims (List[int]): Dimensionality of the hidden layers.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()

        self.encoder = self._make_layers(input_dim, hidden_dims + [latent_dim])
        self.decoder = self._make_layers(latent_dim, hidden_dims[::-1] + [input_dim])
        
        self.sigmoid = nn.Sigmoid()
        
    def _make_layers(self, input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        """Creates the encoder or decoder layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dims (List[int]): List of dimensions for each layer.
        
        Returns:
            nn.Sequential: Sequential container of layers.
        """
        layers = []
        
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.SiLU())  
            input_dim = dim

        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Latent space representation.
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation back to the original input space.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the Autoencoder.

        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            torch.Tensor: Reconstructed data.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        x_hat = self.sigmoid(x_hat)
        return x_hat


def main():
    model = AE(784, [512, 256, 128, 64, 32], 32)
    print(model)

    x = torch.randn(1, 784)
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()