from typing import List
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super(VAE, self).__init__()

        # Build the encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.SiLU())
            prev_dim = h_dim
        
        # Add final layer to split for mean and log variance
        encoder_layers.append(nn.Linear(prev_dim, 2 * latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.SiLU())
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(h_dim, h_dim))
            decoder_layers.append(nn.SiLU())
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # To ensure outputs are between 0 and 1
        self.decoder = nn.Sequential(*decoder_layers)

        # Softplus for stability in KL divergence calculation
        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor, eps: float = 1e-8) -> torch.distributions.MultivariateNormal:
        """
        Encodes input data to latent space, producing distribution over z.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
    
    def reparameterize(self, dist: torch.distributions.MultivariateNormal) -> torch.Tensor:
        """
        Reparameterize to sample from the latent space.
        """
        return dist.rsample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variable z to reconstruct the original input.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the VAE.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        x_recon = self.decode(z)

        # Reconstruction loss (Binary Cross-Entropy)
        loss_recon = F.binary_cross_entropy(x_recon, x + 0.5, reduction='none').sum(-1).mean()

        # KL divergence loss
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        
        # Total loss
        loss = loss_recon + loss_kl

        return x_recon
    
def main():
    model = VAE(input_dim=784, latent_dim=32, hidden_dims=[512, 256, 128, 64, 32])
    print(model)

if __name__ == "__main__":
    main()