import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class DiversifyingMLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)  # Output features for distance comparison
        )

    def forward(self, x):
        return self.net(x)

    def fit(self, holdout_feature, epochs=100, batch_size=32, learning_rate=1e-3):
        """
        Train the VAE on the given holdout features.

        Args:
            holdout_feature: Tensor of shape [holdout_set_size, feature_dim] - the data to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        print("Training MLP")
        # Create DataLoader
        dataset = TensorDataset(holdout_feature)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        criterion = TripletLoss(margin=margin)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                # Assume batch contains triplets: (anchor, positive, negative)
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                # Get features from the pre-trained feature extractor (e.g., ResNet)
                with torch.no_grad():
                    anchor_feats = feature_extractor(anchor)
                    positive_feats = feature_extractor(positive)
                    negative_feats = feature_extractor(negative)

                # Project features using MLP
                anchor_out = mlp(anchor_feats)
                positive_out = mlp(positive_feats)
                negative_out = mlp(negative_feats)

                # Compute loss and update
                loss = criterion(anchor_out, positive_out, negative_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD