# AutoEncoders
This repository provides a collection of autoencoder models, utilities for training and evaluation, and examples for various applications such as image reconstruction, anomaly detection, and feature extraction.

## Getting Started
Install from source
```bash
pip install git+https://github.com/phpfontana/autoencoders.git  
```

```python
import torch
from autoencoders.models import AE
from autoencoders.trainers import Trainer, TrainerConfig

# Define the autoencoder model
model = AE(
    input_dim=784,              # Input dimension (e.g., 28x28 images)
    hidden_dims=[512, 256, 128, 64, 32],  # Hidden layers dimensions
    latent_dim=32,              # Dimensionality of the latent space
)

# Configuration for training
config = TrainerConfig(
    output_dir='experiment_01',     # Directory to save the model and logs
    train_batch_size=64,            # Batch size for training
    val_batch_size=100,             # Batch size for validation
    num_epochs=10,                  # Number of training epochs
    learning_rate=0.0001,           # Initial learning rate
    early_stopping_patience=10,     # Early stopping patience
    optimizer_cls='AdamW',          # Optimizer class (AdamW)
    optimizer_config={'weight_decay': 0.01},  # Optimizer configuration
    scheduler_cls=None,             # Learning rate scheduler class
    scheduler_config=None,          # Learning rate scheduler configuration
)

# Select device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the trainer
trainer = Trainer(
    model=model,
    config=config,
    train_data=train_dataset,  # Replace with your training dataset
    val_data=val_dataset,    # Replace with your validation dataset
    device=device,
)

# Start training
trainer.train()
```