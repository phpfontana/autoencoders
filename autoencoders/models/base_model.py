import torch.nn as nn
from torch import Tensor
from base_config import BaseConfig 

class BaseModel(nn.Module):
    def __init__(
            self,
            config: BaseConfig,
    ):
        super(BaseModel, self).__init__()
        self.config = config
        self.input_shape = config.input_shape
        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError