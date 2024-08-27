from torch import Tensor
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def encode(self, x: Tensor) -> Tensor:
        pass

    def decode(self, z: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass