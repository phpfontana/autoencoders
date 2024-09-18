from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
from .output import VAEOutput


class VAE(nn.Module):
    """Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        hidden_dims (List[int]): Dimensionality of the hidden layers.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()

        self.encoder = self._make_layers(input_dim, hidden_dims + [2 * latent_dim])
        self.decoder = self._make_layers(latent_dim, hidden_dims[::-1] + [input_dim])

        self.softplus = nn.Softplus()

    def _make_layers(self, input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        """Creates the encoder or decoder layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dims (List[int]): List of dimensions for each layer.
        
        Returns:
            nn.Sequential: Sequential container of layers.
        """
        layers = []

        # Add layers with activation functions
        for dim in hidden_dims[:-1]:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.SiLU())
            input_dim = dim

        # Add the final linear layer without activation
        layers.append(nn.Linear(input_dim, hidden_dims[-1]))

        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor, eps: float = 1e-8) -> torch.distributions.MultivariateNormal:
        """Encodes input data to latent space, producing distribution over z.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value for numerical stability.

        Returns:
            torch.distributions.MultivariateNormal: Distribution over z.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)  # Split into mu and logvar
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
    
    def reparameterize(self, dist: torch.distributions.MultivariateNormal) -> torch.Tensor:
        """Reparameterize to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Distribution over z.

        Returns:
            torch.Tensor: Sample from the latent space.
        """
        return dist.rsample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variable z to reconstruct the original input.

        Args:
            z (torch.Tensor): Latent variable.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        x_recon = self.decode(z)
    
        loss_recon = F.mse_loss(x_recon, x, reduction='none').sum(-1).mean()

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )

        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                
        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=x_recon,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )


def main():
    model = VAE(784, [512, 256, 128, 64, 32], 32)
    print(model)

    x = torch.randn(1, 784)
    x = (x - x.min()) / (x.max() - x.min())  
    out = model(x)
    print(out.x_recon.shape)

if __name__ == "__main__":
    main()
