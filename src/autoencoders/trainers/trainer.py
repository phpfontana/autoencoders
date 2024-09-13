import torch
import torch.nn as nn
import os

import torch.utils.data
import torch.utils.tensorboard
import matplotlib.pyplot as plt

from .config import TrainerConfig


class Trainer:
    """Trainer class to train and evaluate the model.

    Args:
        model (nn.Module): The PyTorch model to train.
        config (TrainerConfig): Configuration for the training process.
        train_data (torch.utils.data.Dataset): Dataset for training.
        val_data (torch.utils.data.Dataset): Dataset for validation.
        device (torch.device): The device to run the model (CPU or GPU).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.val_batch_size)
        self.device = device

        self.optimizer = getattr(torch.optim, config.optimizer_cls)(
            model.parameters(), lr=config.learning_rate, **(config.optimizer_config or {})
        )
        
        if config.scheduler_cls:
            self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler_cls)(
                self.optimizer, **(config.scheduler_config or {})
            )
        else:
            self.scheduler = None

        # Initialize TensorBoard writer
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=config.output_dir) if config.output_dir else None

        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)

        self.train_losses = []  # To store training loss values
        self.val_losses = []    # To store validation loss values

    def _train_epoch(self):
        """Train the model for one epoch.

        Returns:
            float: Average loss over the training epoch.
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        
        # Iterate over the training data
        for step, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)

            # Calculate the loss
            loss = outputs.loss

            # Backward pass and optimization step
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update model parameters
            self.optimizer.step()

            total_loss += loss.item()

            # Log training loss every 100 steps
            if step % 100 == 0:
                print(f"Step {step}/{len(self.train_loader)} - loss: {loss.item():.4f}")
                if self.writer:
                    self.writer.add_scalar('Train/Loss', loss.item(), step)

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def _evaluate(self):
        """Evaluate the model on validation data.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0

        with torch.no_grad():
            # Iterate over the validation data
            for data, _ in self.val_loader:
                data = data.to(self.device)

                # Forward pass
                output = self.model(data)

                # Calculate loss
                loss = output.loss
                total_loss += loss.item()

        # Calculate average validation loss
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def _save_plot(self):
        """Save a plot of the training and validation losses to the output directory."""
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Save the loss curve plot
        plot_path = os.path.join(self.config.output_dir, 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss curve saved at {plot_path}")

    def train(self):
        """Train the model for multiple epochs, evaluate on validation data, and save the best and last model."""
        best_loss = float('inf')  # Initialize best loss to infinity

        for epoch in range(self.config.num_epochs):            
            # Train for one epoch
            train_loss = self._train_epoch()

            # Evaluate on the validation data
            val_loss = self._evaluate()

            # Update the learning rate scheduler, if used
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.config.num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            # Log average losses to TensorBoard, if enabled
            if self.writer:
                self.writer.add_scalar('Train/AvgLoss', train_loss, epoch)
                self.writer.add_scalar('Val/AvgLoss', val_loss, epoch)

            # Save the best model based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = os.path.join(self.config.output_dir, 'best.pt')
                torch.save(self.model.state_dict(), best_model_path)

        # Save the final model at the end of training
        last_model_path = os.path.join(self.config.output_dir, 'last.pt')
        torch.save(self.model.state_dict(), last_model_path)

        # Save the loss plot (training vs validation losses)
        self._save_plot()