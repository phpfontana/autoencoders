from autoencoders.models import AutoEncoder

def main():
    model = AutoEncoder(input_dim=784, latent_dim=2)
    print(model)

if __name__ == '__main__':
    main()