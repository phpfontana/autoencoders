from torch import Tensor
from ..base_model import BaseModel

class VAE(BaseModel):
    def __init__(self):
        super(VAE, self).__init__()

    def encode(self, x: Tensor) -> Tensor:
        pass

    def decode(self, z: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass