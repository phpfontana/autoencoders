from dataclasses import dataclass

import torch


@dataclass
class AEOutput:
    """Dataclass for AE output
    Attributes:
        z (torch.Tensor): Latent space representation.
        x_recon (torch.Tensor): Reconstructed data.
        loss (torch.Tensor): Loss value.
    """

    z: torch.Tensor
    x_recon: torch.Tensor

    loss: torch.Tensor