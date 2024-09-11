from dataclasses import dataclass
from typing import Union, Optional, Dict

@dataclass
class TrainerConfig:
    """Dataclass for Trainer configuration.

    Attributes:
        output_dir (Union[str, None]): Output directory to save the model, logs, and plots.
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_cls (str): Optimizer class to use (e.g. AdamW).
        optimizer_config (Optional[Dict]): Additional configuration for the optimizer.
        scheduler_cls (Optional[str]): Scheduler class to use for learning rate scheduling.
        scheduler_config (Optional[Dict]): Additional configuration for the scheduler.
        loss_fn (str): Loss function to use (e.g. BCELoss).
    """
    output_dir: Optional[str] = None
    train_batch_size: int = 64
    val_batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 2e-5
    optimizer_cls: str = 'AdamW'
    optimizer_config: Optional[Dict] = None
    scheduler_cls: Optional[str] = None
    scheduler_config: Optional[Dict] = None
    loss_fn: str = 'BCELoss'
