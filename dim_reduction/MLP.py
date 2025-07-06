import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from pytorch_metric_learning import losses
from torchvision import models, transforms
from PIL import Image



device = "cuda"

# MLP for dimensionality reduction
class DiversifyingMLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[3].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

    def fit(self, features, labels, epochs=100, batch_size=32, learning_rate=1e-3, margin=0.2):
        """
        Train with triplet loss.

        Args:
            features: Tensor of ResNet features [n_samples, feature_dim]
            labels: Tensor of labels [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size for triplet training
            learning_rate: Learning rate for Adam optimizer
            margin: Margin for triplet loss
        """
        self.to(device)
        features = features.to(device)
        labels = labels.to(device)

        criterion = losses.TripletMarginLoss(margin=margin)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Select semi-hard triplets
        def select_semi_hard_triplets(features, labels, num_triplets=1000):
            triplets = []
            for i in range(len(features)):
                anchor = features[i]
                anchor_label = labels[i]
                pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=device) != i)
                neg_mask = (labels != anchor_label)
                pos_indices = torch.where(pos_mask)[0]
                neg_indices = torch.where(neg_mask)[0]
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    pos_dist = torch.norm(features[pos_indices] - anchor, dim=1)
                    neg_dist = torch.norm(features[neg_indices] - anchor, dim=1)
                    valid_pos = pos_indices[pos_dist > torch.min(neg_dist) + margin]
                    valid_neg = neg_indices[neg_dist < torch.max(pos_dist) - margin]
                    if len(valid_pos) > 0 and len(valid_neg) > 0:
                        hardest_pos = valid_pos[torch.argmax(pos_dist[valid_pos])]
                        hardest_neg = valid_neg[torch.argmin(neg_dist[valid_neg])]
                        triplets.append((i, hardest_pos, hardest_neg))
            return triplets[:min(num_triplets, len(triplets))]

        triplets = select_semi_hard_triplets(features, labels)
        if not triplets:
            raise ValueError("No valid triplets found. Check holdout dataset labels.")

        triplet_dataset = TensorDataset(
            torch.tensor([features[i] for i, _, _ in triplets], dtype=torch.float32),
            torch.tensor([features[j] for _, j, _ in triplets], dtype=torch.float32),
            torch.tensor([features[k] for _, _, k in triplets], dtype=torch.float32)
        )
        dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for anchor, positive, negative in dataloader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out = self(anchor)
                positive_out = self(positive)
                negative_out = self(negative)
                loss = criterion(anchor_out, positive_out, negative_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    def fit_transform(self, cal_feature, test_feature):
        """
        Transform calibration and test features using trained MLP.

        Args:
            cal_feature: Tensor of calibration features [n_cal, feature_dim]
            test_feature: Tensor of test feature [feature_dim]
        Returns:
            cal_reduced: Reduced calibration features [n_cal, output_dim]
            test_reduced: Reduced test feature [output_dim]
        """
        self.eval()
        with torch.no_grad():
            cal_reduced = self(cal_feature.to(device))
            test_reduced = self(test_feature.to(device))
        return cal_reduced, test_reduced
