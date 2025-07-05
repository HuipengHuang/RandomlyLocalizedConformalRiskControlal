import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class SparseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, sparsity_weight=0.1):
        super(SparseVAE, self).__init__()
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight  # Weight for L1 sparsity penalty

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 512)
        self.fc3 = nn.Linear(512, input_dim)

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
        return self.decode(z), mu, logvar, z  # Also return z for sparsity penalty

    def loss_function(self, recon_x, x, mu, logvar, z):
        # Reconstruction loss (MSE or BCE)
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
        # reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence (Gaussian prior)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Sparsity penalty (L1 on latent activations)
        sparsity_penalty = self.sparsity_weight * torch.sum(torch.abs(z))

        # Total loss
        total_loss = reconstruction_loss + kl_divergence + sparsity_penalty
        return total_loss

    def fit(self, holdout_feature, epochs=100, batch_size=32, learning_rate=1e-3):
        print("Training SVAE")
        dataset = TensorDataset(holdout_feature)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for batch_features, in dataloader:
                # Forward pass (return z for sparsity penalty)
                recon_batch, mu, logvar, z = self(batch_features)

                # Compute loss
                loss = self.loss_function(recon_batch, batch_features, mu, logvar, z)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("Finish Training SVAE")

