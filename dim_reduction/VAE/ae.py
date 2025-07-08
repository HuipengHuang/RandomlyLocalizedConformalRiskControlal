import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.input_dim = input_dim
        print(f"input dimension:{input_dim}")

        # Encoder
        """self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)  # Directly output latent representation
        )

        # Decoder (no final activation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, input_dim)
        )"""

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder (no final activation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
    def encode(self, x):
        return self.encoder(x)

    def fit_transform(self, cal_feature, test_feature):
        new_cal_feature = self.encode(cal_feature)
        new_test_feature = self.encode(test_feature)
        return new_cal_feature, new_test_feature

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def add_noise(self, x):
        return x + torch.randn_like(x) * 0.1

    def fit(self, holdout_feature, epochs=300, batch_size=32, learning_rate=1e-3):
        train_loader = DataLoader(TensorDataset(holdout_feature),
                                  batch_size=batch_size,
                                  shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                batch = batch[0]
                #noisy_batch = self.add_noise(batch)
                noisy_batch = batch
                recon = self.forward(noisy_batch)

                # MSE reconstruction loss
                loss = criterion(recon, batch)

                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()