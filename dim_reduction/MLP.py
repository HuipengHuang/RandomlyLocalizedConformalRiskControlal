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
        log_prob = torch.log(exp_sim * pos_mask + 1e-6) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        # Sum over positives
        loss = -log_prob[pos_mask].sum() / pos_mask.sum()

        return loss


class NPairLoss(nn.Module):
    """N-Pair Loss with L2 distance (Euclidean distance)."""

    def __init__(self, temperature=1e-1):
        super().__init__()
        self.temperature = temperature

    class NPairLoss(nn.Module):
        """N-Pair Loss with L2 distance (Euclidean distance)."""

        def __init__(self, temperature=1e-1):
            super().__init__()
            self.temperature = temperature

        def forward(self, embeddings, labels):
            """
            Args:
                embeddings: Feature vectors [batch_size, feature_dim]
                labels: Ground truth labels [batch_size]
            Returns:
                loss: PyTorch tensor with gradient tracking
            """
            device = embeddings.device
            batch_size = embeddings.shape[0]

            # Compute pairwise squared L2 distances
            dist_matrix = torch.cdist(embeddings, embeddings, p=2).pow(2)  # [batch_size, batch_size]

            # Convert distance to similarity
            sim_matrix = -dist_matrix / self.temperature

            # Create mask for positive pairs
            pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
            pos_mask.fill_diagonal_(False)

            # Initialize loss components
            losses = []
            valid_pairs = 0

            for i in range(batch_size):
                pos_indices = torch.where(pos_mask[i])[0]
                if len(pos_indices) == 0:
                    continue

                j = pos_indices[torch.randint(0, len(pos_indices), (1,))].item()
                neg_mask = labels != labels[i]

                if not neg_mask.any():
                    continue

                numerator = sim_matrix[i, j]
                denominator = torch.logsumexp(sim_matrix[i, neg_mask], dim=0)
                losses.append(- (numerator - denominator))
                valid_pairs += 1

            if valid_pairs == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            # Stack all individual losses and compute mean
            loss = torch.stack(losses).mean()
            return loss

class DiversifyingMLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[2].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

    def fit(self, features, labels, epochs=100, batch_size=32, learning_rate=1e-3, loss_type="n_pair", temperature=0.1):
        """
        Train with either N-Pair Loss or SupCon Loss.

        Args:
            loss_type: "n_pair" or "supcon"
            temperature: Softmax temperature parameter
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features = features.to(device)
        labels = labels.to(device)

        # Initialize loss
        if loss_type == "n_pair":
            criterion = NPairLoss(temperature=temperature)
        elif loss_type == "supcon":
            criterion = SupConLoss(temperature=temperature)
        else:
            raise ValueError("loss_type must be 'n_pair' or 'supcon'")

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        dataloader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                embeddings = self(batch_features)
                embeddings = F.normalize(embeddings, dim=1)  # L2-normalize for contrastive loss

                # Compute loss
                loss = criterion(embeddings, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    def fit_transform(self, cal_feature, test_feature):
        return self(cal_feature), self(test_feature)



