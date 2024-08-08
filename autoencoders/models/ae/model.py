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
from ..base import BaseModel
from typing import Tuple, Any, Union, List

class AE(BaseModel):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], # (batch_size, channels, height, width)
                 hidden_dims: List[int], 
                 latent_dim: int): 
        super(AE, self).__init__()
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Flatten input size
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]

        # Encoder layers
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = h_dim
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        decoder_input_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(decoder_input_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_input_dim = h_dim
        decoder_layers.append(nn.Linear(decoder_input_dim, self.input_shape[0] * self.input_shape[1] * self.input_shape[2]))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        x_hat = self.decoder(z)
        # Reshape to the original input shape
        return x_hat.view(x_hat.size(0), *self.input_shape)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
