import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss (Khosla et al., 2020)"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Normalized feature vectors [batch_size, feature_dim]
            labels: Ground truth labels [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # Compute similarity matrix
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positives (same class, excluding self)
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [batch_size, batch_size]
        pos_mask = label_matrix.fill_diagonal_(False)  # Exclude self-comparison

        # Compute log-softmax
        exp_sim = torch.exp(sim_matrix)
        log_prob = torch.log(exp_sim * pos_mask) - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Sum over positives
        loss = -log_prob[pos_mask].sum() / pos_mask.sum()

        return loss


class DiversifyingMLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=10):  # Increased output dim for better separation
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

    def fit(self, features, labels, epochs=100, batch_size=32, learning_rate=1e-3, temperature=0.1):
        """
        Train with Supervised Contrastive Learning.

        Args:
            features: Tensor of shape [N, feature_dim]
            labels: Tensor of shape [N] with class indices
            temperature: Softmax temperature parameter
        """
        device = "cuda"
        features = features.to(device)
        labels = labels.to(device)

        # Create DataLoader
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize loss and optimizer
        criterion = SupConLoss(temperature=temperature)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                embeddings = self(batch_features)

                # Compute loss
                loss = criterion(embeddings, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



