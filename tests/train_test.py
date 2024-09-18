from autoencoders.models import AE, VAE
from autoencoders.trainers import Trainer, TrainerConfig
from torch.utils.tensorboard import summary
import torch
import torchvision

def main():

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)    
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)   
    model = VAE(input_dim=784, latent_dim=32, hidden_dims=[128, 64])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = TrainerConfig(
        output_dir='experiment_02',
        train_batch_size=128,
        val_batch_size=128,
        num_epochs=50,
        learning_rate=1e-3,
        optimizer_cls='AdamW',
        optimizer_config={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_config={"patience": 5, "factor": 0.5},
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(
        model=model,
        config=config,
        train_data=train_dataset,
        val_data=val_dataset,
        device=device,
    )

    trainer.train()
    
if __name__ == '__main__':
    main()
