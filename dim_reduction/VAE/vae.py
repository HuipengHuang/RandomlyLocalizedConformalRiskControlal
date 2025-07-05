
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.input_dim = input_dim
        print(f"input dimension:{input_dim}")
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder (no final activation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # Store normalization parameters
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))

    def normalize(self, x):
        return (x - self.feature_mean) / (self.feature_std + 1e-8)

    def denormalize(self, x):
        return x * self.feature_std + self.feature_mean

    def encode(self, x):
        h = self.encoder(self.normalize(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def fit_transform(self, cal_feature, test_feature):
        new_cal_feature, _ = self.encode(cal_feature)
        new_test_feature, _ = self.encode(test_feature)
        return new_cal_feature, new_test_feature

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.denormalize(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def add_noise(self, x):
        return x + torch.randn_like(x) * 1

    def fit(self, holdout_feature, epochs=300, batch_size=32, learning_rate=1e-3):
        # Calculate and store normalization parameters
        self.feature_mean = holdout_feature.mean(dim=0)
        self.feature_std = holdout_feature.std(dim=0)

        normalized_features = self.normalize(holdout_feature)


        train_loader = DataLoader(TensorDataset(normalized_features), batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                batch = batch[0]
                recon, mu, logvar = self.forward(self.denormalize(self.add_noise(batch)))  # Decoder outputs denormalized

                # MSE on original scale
                recon_loss = F.mse_loss(recon, self.denormalize(batch), reduction='sum')

                # KL divergence
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + kl_div

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()






