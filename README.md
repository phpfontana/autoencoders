# AutoEncoders
This repository provides a collection of autoencoder models, utilities for training and evaluation, and examples for various applications such as image reconstruction, anomaly detection, and feature extraction.

```python
import torch
from autoencoders import Autoencoder, Trainer, TrainerConfig

# Define the autoencoder model
model = Autoencoder(
    input_dim=784,              # Input dimension (e.g., 28x28 images)
    latent_dim=32,              # Dimensionality of the latent space
    hidden_dims=[512, 256, 128, 64, 32]  # Hidden layers dimensions
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
    loss_fn='BCELoss',              # Loss function
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