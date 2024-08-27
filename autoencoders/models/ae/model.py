import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class BasicBlock(nn.Module):
    """
    A basic neural network block consisting of a linear layer followed by an activation function.
    """
    def __init__(self, input_dim: int, output_dim: int, activation: str = "SiLU"):
        """
        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            activation (str): The name of the activation function to use. Default is 'SiLU'.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the BasicBlock.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor after applying the linear layer and activation.
        """
        return self.activation(self.fc(x))


class Encoder(nn.Module):
    """
    The encoder part of the Autoencoder, mapping the input to a latent space representation.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        """
        Args:
            input_dim (int): The number of input features.
            hidden_dims (List[int]): A list containing the number of units in each hidden layer.
            latent_dim (int): The number of units in the latent space representation.
        """
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(BasicBlock(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(BasicBlock(hidden_dims[i-1], hidden_dims[i]))
        
        # Latent layer
        layers.append(BasicBlock(hidden_dims[-1], latent_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Encoder.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Latent space representation.
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    The decoder part of the Autoencoder, reconstructing the input from the latent space representation.
    """
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Args:
            latent_dim (int): The number of units in the latent space representation.
            hidden_dims (List[int]): A list containing the number of units in each hidden layer.
            output_dim (int): The number of output features (should match input_dim of Encoder).
        """
        super().__init__()
        layers = []
        
        # Latent layer
        layers.append(BasicBlock(latent_dim, hidden_dims[-1]))
        
        # Hidden layers in reverse order
        for i in range(len(hidden_dims) - 1, 0, -1):
            layers.append(BasicBlock(hidden_dims[i], hidden_dims[i-1]))
        
        # Output layer
        layers.append(BasicBlock(hidden_dims[0], output_dim, activation="Sigmoid"))
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass through the Decoder.
        
        Args:
            z (Tensor): Latent space representation.
            
        Returns:
            Tensor: Reconstructed input tensor.
        """
        return self.decoder(z)


class AE(nn.Module):
    """
    An Autoencoder model comprising an Encoder and a Decoder.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        """
        Args:
            input_dim (int): The number of input features.
            latent_dim (int): The number of units in the latent space representation.
            hidden_dims (List[int]): A list containing the number of units in each hidden layer.
        """
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        
    def encode(self, x: Tensor) -> Tensor:
        """
        Encodes the input into the latent space.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Latent space representation.
        """
        return self.encoder(x)
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes the latent space representation back to the input space.
        
        Args:
            z (Tensor): Latent space representation.
            
        Returns:
            Tensor: Reconstructed input tensor.
        """
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Autoencoder (encode -> decode).
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Reconstructed input tensor.
        """
        z = self.encode(x)
        return self.decode(z)


def main():
    """
    Main function for testing the Autoencoder model.
    """
    # Example input tensor with batch size 1 and 784 features
    x = torch.randn(1, 784)
    
    # Hidden layer dimensions
    hidden_dims = [256, 128, 64, 32]
    
    # Instantiate Autoencoder model
    ae = Autoencoder(input_dim=784, latent_dim=16, hidden_dims=hidden_dims)
    
    # Print model architecture
    print(ae)

    # Forward pass to obtain reconstructed output
    x_hat = ae(x)
    print(f"Reconstructed shape: {x_hat.shape}")

if __name__ == "__main__":
    main()