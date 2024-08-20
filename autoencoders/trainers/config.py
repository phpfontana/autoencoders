import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Union, Optional

@dataclass
class TrainerConfig:
    output_dir: Union[str, None] = None
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    checkpoint_dir: str = 'checkpoints'  # Directory to save checkpoints
    early_stopping_patience: int = 10  # Patience for early stopping
    optimizer_type: str = 'adamw'  # Type of optimizer ('adamw', 'sgd', etc.)
    loss_fn_type: str = 'mse'  # Loss function type ('mse', 'cross_entropy', etc.)
