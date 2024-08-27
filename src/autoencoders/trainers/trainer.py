import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from .config import TrainerConfig


class Trainer:
    """Trainer class to train and evaluate the model

    Args:
        model (nn.Module): Model to train
        config (TrainerConfig): Trainer configuration
        train_data (Dataset): Training data
        val_data (Dataset): Validation data
        device (torch.device): Device to use for training
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainerConfig,
            train_data: Dataset,
            val_data: Dataset,
            device: torch.device,
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

        self.model.to(self.device)

        self.optimizer = self._init_optimizer()
        self.loss_fn = self._init_loss_fn()
        self.scheduler = self._init_scheduler()

        self.train_loader = self._init_dataloader(train_data, config.train_batch_size, shuffle=True)
        self.val_loader = self._init_dataloader(val_data, config.val_batch_size, shuffle=False)

        os.makedirs(self.config.output_dir, exist_ok=True)

    def _init_optimizer(self):
        optimizer_cls = getattr(torch.optim, self.config.optimizer_cls)

        if self.config.optimizer_config is not None:
            return optimizer_cls(self.model.parameters(), lr=self.config.learning_rate, **self.config.optimizer_config)
        else:
            return optimizer_cls(self.model.parameters(), lr=self.config.learning_rate)

    def _init_scheduler(self):
        if self.config.scheduler_cls is not None:
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.config.scheduler_cls)
            return scheduler_cls(self.optimizer, **self.config.scheduler_config)
        else:
            return None

    def _init_loss_fn(self):
        return getattr(nn, self.config.loss_fn)()

    def _init_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def _train(self):
        """Train the model on batch"""
        self.model.train()
        running_loss = 0.0

        for i, (train_batch, labels_batch) in enumerate(self.train_loader):
            train_batch = train_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)

            output_batch = self.model(train_batch)
            loss = self.loss_fn(output_batch, labels_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _evaluate(self):
        """Evaluate the model on batch"""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for i, (val_batch, labels_batch) in enumerate(self.val_loader):
                val_batch = val_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                output_batch = self.model(val_batch)
                loss = self.loss_fn(output_batch, labels_batch)

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def train(self):
        """Train and evaluate the model"""
        best_val_loss = float('inf')
        patience = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config.num_epochs):
            train_loss = self._train()
            val_loss = self._evaluate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            save_model(self.model, os.path.join(self.config.output_dir, 'last.pt'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(self.model, os.path.join(self.config.output_dir, 'best.pt'))
                patience = 0
            else:
                patience += 1

            # Early stopping
            if patience > self.config.early_stopping_patience:
                print("Early stopping")
                break
        
        # save plot fig        
        plt.plot(train_losses, label='train_loss')  # Use train_losses list
        plt.plot(val_losses, label='val_loss')      # Use val_losses list
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.config.output_dir, 'loss.png'))
        plt.close()  # Close the figure to free up memory

        
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss