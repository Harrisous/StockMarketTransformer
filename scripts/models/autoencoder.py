import torch.nn as nn
class Autoencoder(nn.Module):
    '''Autoencode class to extract information for tickers and indices'''
    def __init__(self, input_dim, latent_dim, size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128*size),
            nn.ReLU(True),
            nn.Linear(128*size, 64*size),
            nn.ReLU(True),
            nn.Linear(64*size, latent_dim) # Compresses to latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*size),
            nn.ReLU(True),
            nn.Linear(64*size, 128*size),
            nn.ReLU(True),
            nn.Linear(128*size, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded