import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .config import TrainerConfig


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            config: TrainerConfig,
            train_data: DataLoader,
            val_data: DataLoader,
            device: torch.device,
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

        self.model.to(self.device)

        # Initialize the optimizer and loss function based on the config
        self.optimizer = self._initialize_optimizer()
        self.loss_fn = self._initialize_loss_fn()

        # Create checkpoint directory if it does not exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on the config"""
        if self.config.optimizer_type == 'adamw':
            return AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        # Add more optimizers here as needed
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")

    def _initialize_loss_fn(self) -> nn.Module:
        """Initialize loss function based on the config"""
        if self.config.loss_fn_type == 'mse':
            return nn.MSELoss()
        elif self.config.loss_fn_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config.loss_fn_type == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss function type: {self.config.loss_fn_type}")

    def _train(self):
        """Train the model on batch"""
        self.model.train()
        running_loss = 0.0

        for i, (train_batch, _) in enumerate(self.train_data):
            train_batch = train_batch.to(self.device)

            output_batch = self.model(train_batch)
            loss = self.loss_fn(output_batch, train_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_data)

    def _evaluate(self):
        """Evaluate the model on batch"""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for i, (val_batch, _) in enumerate(self.val_data):
                val_batch = val_batch.to(self.device)

                output_batch = self.model(val_batch)
                loss = self.loss_fn(output_batch, val_batch)

                running_loss += loss.item()

        return running_loss / len(self.val_data)

    def train(self):
        """Train and evaluate the model"""
        best_val_loss = float('inf')
        patience = 0

        for epoch in range(self.config.num_epochs):
            train_loss = self._train()
            val_loss = self._evaluate()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            # Save the model if the validation loss is the best we've seen so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, 'best_model.pt'))
                patience = 0
            else:
                patience += 1

            # Early stopping
            if patience > self.config.early_stopping_patience:
                print("Early stopping")
                break
