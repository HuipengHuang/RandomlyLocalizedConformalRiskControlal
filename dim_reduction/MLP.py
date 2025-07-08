import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized feature vectors [batch_size, feature_dim]
            labels: Ground truth labels [batch_size]
        Returns:
            loss: Triplet loss with hard mining
        """
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # L2 distance matrix
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).bool()

        if self.hard_mining:
            # Hardest positive (furthest within class)
            pos_dist = pairwise_dist[pos_mask].view(len(labels), -1).max(dim=1)[0]

            # Hardest negative (closest across classes)
            neg_dist = pairwise_dist + (~neg_mask).float() * 1e6  # Mask non-negatives
            neg_dist = neg_dist.min(dim=1)[0]
        else:
            # Random positive/negative (slower but more stable)
            pos_dist = pairwise_dist[pos_mask].view(len(labels), -1).mean(dim=1)
            neg_dist = pairwise_dist[neg_mask].view(len(labels), -1).mean(dim=1)

        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss

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

    def __init__(self, temperature=0.1):
        super(NPairLoss, self).__init__()
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

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # Create mask for positive pairs (same class, excluding self)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
        pos_mask.fill_diagonal_(False)

        # Initialize loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_pairs = 0

        for i in range(batch_size):
            # Get positive samples
            pos_indices = torch.where(pos_mask[i])[0]
            if len(pos_indices) == 0:
                continue

            # Randomly select one positive
            j = pos_indices[torch.randint(0, len(pos_indices), (1,))]

            # Get negative samples
            neg_mask = (labels != labels[i])
            if not neg_mask.any():
                continue

            # Compute loss for this pair
            numerator = sim_matrix[i, j]
            denominator = torch.logsumexp(sim_matrix[i, neg_mask], dim=0)
            loss = loss + (- (numerator - denominator))
            valid_pairs += 1

        if valid_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss / valid_pairs


class ContrastiveMSELoss(nn.Module):
    def __init__(self, margin=1.0, same_class_weight=1.0, diff_class_weight=1.0):
        super(ContrastiveMSELoss, self).__init__()
        self.margin = margin  # Minimum desired distance between different classes
        self.same_class_weight = same_class_weight  # Weight for same-class penalty
        self.diff_class_weight = diff_class_weight  # Weight for different-class penalty

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)
        Returns:
            loss: Contrastive MSE loss
        """
        batch_size = embeddings.size(0)

        # Compute pairwise L2 distances (Euclidean distance)
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # [batch_size, batch_size]

        # Create mask for same-class and different-class pairs
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # [batch_size, batch_size]
        same_class_mask = torch.triu(label_matrix, diagonal=1)  # Upper triangular to avoid duplicates
        diff_class_mask = torch.triu(~label_matrix, diagonal=1)  # Upper triangular for different classes

        # Get distances for same-class and different-class pairs
        same_class_dist = pairwise_dist[same_class_mask]
        diff_class_dist = pairwise_dist[diff_class_mask]

        # Loss for same-class pairs (minimize distance → MSE)
        same_class_loss = torch.mean(same_class_dist ** 2) if len(same_class_dist) > 0 else 0.0

        # Loss for different-class pairs (maximize distance → hinge-like penalty)
        diff_class_loss = torch.mean(F.relu(self.margin - diff_class_dist) ** 2) if len(diff_class_dist) > 0 else 0.0

        # Total loss (weighted combination)
        loss = (self.same_class_weight * same_class_loss) - (self.diff_class_weight * diff_class_loss)

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

    def fit(self, features, labels, epochs=100, batch_size=32,
            learning_rate=1e-3, loss_type="mse", margin=1.0, temperature=0.1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # Initialize loss
        if loss_type == "triplet":
            criterion = TripletMarginLoss(margin=margin)
        elif loss_type == "n_pair":
            criterion = NPairLoss(temperature=temperature)
        elif loss_type == "supcon":
            criterion = SupConLoss(temperature=temperature)
        elif loss_type == "mse":
            criterion = ContrastiveMSELoss()
        else:
            raise ValueError("loss_type must be 'triplet', 'n_pair', or 'supcon'")

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        dataloader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass + L2 normalization
                embeddings = self(batch_features)

                loss = criterion(embeddings, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def fit_transform(self, cal_feature, test_feature):
        return self(cal_feature), self(test_feature)
