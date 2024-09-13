from autoencoders.models import AE, VAE
from autoencoders.trainers import Trainer, TrainerConfig

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2

def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = VAE(input_dim=784, latent_dim=32, hidden_dims=[512, 256, 128, 64, 32])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the trained model with weights_only=True to mitigate the security risk
    model.load_state_dict(torch.load('experiment_02/best.pt', weights_only=True))
    model.eval()

    # Number of examples to visualize
    num_examples = 5

    # Get a batch of validation data
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=num_examples, shuffle=True)
    data_iter = iter(val_loader)

    # Use Python's built-in next() to get the next batch
    images, _ = next(data_iter)
    images = images.to(device)

    # Pass the images through the model to get reconstructed images
    with torch.no_grad():
        output = model(images)  # Get AEOutput object
        reconstructed = output.x_recon.cpu()  # Move the reconstructed images (x_recon) to CPU

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 4))
    
    # Plot original images
    for i in range(num_examples):
        img = images[i].cpu().view(28, 28)  # Reshape the flattened image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

    # Plot reconstructed images
    for i in range(num_examples):
        img = reconstructed[i].view(28, 28)  # Reshape the flattened image
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')

    plt.show()

if __name__ == '__main__':
    main()
