import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class AE(nn.Module):
    def __init__(
            self, 
            input_dim: int,  
            latent_dim: int, 
            hidden_dims: List[int] = None
    ):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.SiLU())
        for i in range(1, len(hidden_dims)):
            encoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.SiLU())
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.SiLU())
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    

def main():
    x = torch.randn(1, 784)
    hidden_dims = [256, 128, 64, 32]
    ae = AE(input_dim=784, latent_dim=16, hidden_dims=hidden_dims)
    print(ae)

    x_hat = ae(x)
    print(x_hat.shape)

if __name__ == "__main__":
    main()
