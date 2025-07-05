import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VariationalAutoEncoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 512)
        self.fc3 = nn.Linear(512, input_dim)

    def fit_transform(self, cal_feature, test_feature):
        new_cal_feature, _ = self.encode(cal_feature)
        new_test_feature, _ = self.encode(test_feature)
        return new_cal_feature, new_test_feature

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


    def fit(self, holdout_feature, epochs=100, batch_size=32, learning_rate=1e-3):
        """
        Train the VAE on the given holdout features.

        Args:
            holdout_feature: Tensor of shape [holdout_set_size, feature_dim] - the data to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        print("Training VAE")
        # Create DataLoader
        dataset = TensorDataset(holdout_feature)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_features, in dataloader:  # Unpack the batch
                # Forward pass
                reconstructed, mu, logvar = self(batch_features)

                # Compute loss
                #reconstruction_loss = F.binary_cross_entropy(
                 #   reconstructed, batch_features, reduction='sum'
                #)
                reconstruction_loss = F.mse_loss(
                    reconstructed,
                    batch_features,
                    reduction='sum'  # Sum over batch and features
                )

                # KL divergence
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # Total loss
                loss = reconstruction_loss + kl_divergence

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("Finish Training VAE")


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
