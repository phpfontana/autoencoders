import torch
import torch.nn as nn
from typing import List, Tuple

class BaseConfig:
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 latent_dim: int,
                 hidden_dims: List[int],
                ):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        