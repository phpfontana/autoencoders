import torch.nn as nn
from torch import Tensor

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError