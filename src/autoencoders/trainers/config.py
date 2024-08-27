from dataclasses import dataclass
from typing import Union, Optional, Dict

@dataclass
class TrainerConfig:
    output_dir: Union[str, None] = None
    train_batch_size: int = 64
    val_batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 2e-5
    early_stopping_patience: int = 10  
    optimizer_cls: str = 'AdamW'  
    optimizer_config: Optional[Dict] = None
    scheduler_cls: Optional[str] = None
    scheduler_config: Optional[Dict] = None
    loss_fn: str = 'BCELoss'  


def main():
    config = TrainerConfig(
        output_dir='output',
        train_batch_size=64,
        val_batch_size=100,
        num_epochs=100,
        learning_rate=0.0001,
        early_stopping_patience=10,
        optimizer_cls='AdamW',
        optimizer_config={'weight_decay': 0.01},
        scheduler_cls=None,
        scheduler_config=None,
        loss_fn='BCELoss',
    )

    print(config)

if __name__ == '__main__':
    main()