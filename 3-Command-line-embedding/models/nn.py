import os

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """
    AutoEncoder model for command line embeddings compression.
    """
    model_save_name = 'pytorch_model.pt'

    def __init__(self, embed_dim=384, encoder_dim=16):
        """
        param: embed_dim: int: Dimension of the input embeddings
        param: encoder_dim: int: Dimension of the encoder output
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        b_size = x.shape[0]
        x = x.view(b_size, 1, -1)
        x = self.encode(x)
        x = self.decode(x)
        x = x.view(b_size, -1)
        return x

    def save(self, path: str):
        path = os.path.join(path, self.model_save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_params = {
            'embed_dim': self.embed_dim,
            'encoder_dim': self.encoder_dim,
            'state_dict': self.state_dict()
        }
        torch.save(model_params, path)

    @staticmethod
    def load(path: str):
        path = os.path.join(path, AutoEncoder.model_save_name)
        model_params = torch.load(path)
        model = AutoEncoder(embed_dim=model_params['embed_dim'], encoder_dim=model_params['encoder_dim'])
        model.load_state_dict(model_params['state_dict'])
        return model


