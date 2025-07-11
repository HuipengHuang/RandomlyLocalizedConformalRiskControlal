import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from typing import Tuple


class NormalizingFlow(nn.Module):
    def __init__(self, input_dim: int, n_flows: int = 4, hidden_dim: int = 64):
        """
        Normalizing Flow model using affine coupling layers.

        Args:
            input_dim: Dimension of input data
            n_flows: Number of flow transformations
            hidden_dim: Hidden dimension for neural networks in coupling layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_flows = n_flows

        # Create a sequence of flow transformations
        self.transforms = nn.ModuleList([
            AffineCoupling(input_dim, hidden_dim)
            for _ in range(n_flows)
        ])

        # Base distribution is standard normal
        self.register_buffer('base_dist_mean', torch.zeros(input_dim))
        self.register_buffer('base_dist_std', torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map input data to latent space and compute log determinant of Jacobian.

        Args:
            x: Input data tensor

        Returns:
            z: Latent representation
            log_det_jac: Log determinant of Jacobian
        """
        log_det_jac = 0
        z = x

        for transform in self.transforms:
            z, ldj = transform.forward(z)
            log_det_jac += ldj

        return z, log_det_jac

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map latent representation back to data space.

        Args:
            z: Latent representation

        Returns:
            x: Reconstructed data
            log_det_jac: Log determinant of Jacobian
        """
        log_det_jac = 0
        x = z

        for transform in reversed(self.transforms):
            x, ldj = transform.inverse(x)
            log_det_jac += ldj

        return x, log_det_jac

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of input data under the model.

        Args:
            x: Input data tensor

        Returns:
            log_prob: Log probability of input data
        """
        z, log_det_jac = self.forward(x)

        # Compute log probability under base distribution (standard normal)
        log_prob_z = torch.distributions.Normal(
            self.base_dist_mean, self.base_dist_std
        ).log_prob(z).sum(dim=1)

        # Apply change of variables formula
        log_prob_x = log_prob_z + log_det_jac
        return log_prob_x

    def fit_transform(self, cal_feature: torch.Tensor, test_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform calibration and test features to latent space.

        Args:
            cal_feature: Calibration features
            test_feature: Test features

        Returns:
            Transformed calibration and test features
        """
        with torch.no_grad():
            new_cal_feature, _ = self.forward(cal_feature)
            new_test_feature, _ = self.forward(test_feature)
        return new_cal_feature, new_test_feature

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            n_samples: Number of samples to generate

        Returns:
            samples: Generated samples
        """
        with torch.no_grad():
            # Sample from base distribution
            z = torch.randn(n_samples, self.input_dim).to(self.base_dist_mean.device)

            # Transform through inverse flow
            samples, _ = self.inverse(z)

        return samples

    def fit(self, holdout_feature: torch.Tensor, epochs: int = 300,
            batch_size: int = 32, learning_rate: float = 1e-3):
        """
        Train the normalizing flow model.

        Args:
            holdout_feature: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        train_loader = DataLoader(TensorDataset(holdout_feature),
                                  batch_size=batch_size,
                                  shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch in train_loader:
                batch = batch[0]

                # Compute negative log likelihood
                log_prob = self.log_prob(batch)
                loss = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')


class AffineCoupling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        assert input_dim >= 2, "Input dim must be ≥ 2 for splitting."
        self.input_dim = input_dim
        self.split_dim = input_dim // 2

        # Network outputs parameters for the *second* split (must match dims)
        self.net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (input_dim - self.split_dim) * 2)  # Final dim matches!
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize last layer to output zeros (identity transform)
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x[:, :self.split_dim], x[:, self.split_dim:]
        params = self.net(x_a)
        log_scale, shift = params.chunk(2, dim=1)
        z_b = x_b * torch.exp(log_scale) + shift
        z = torch.cat([x_a, z_b], dim=1)
        log_det_jac = log_scale.sum(dim=1)
        return z, log_det_jac

    def inverse(self, z):
        z_a, z_b = z[:, :self.split_dim], z[:, self.split_dim:]
        params = self.net(z_a)
        log_scale, shift = params.chunk(2, dim=1)
        x_b = (z_b - shift) * torch.exp(-log_scale)
        x = torch.cat([z_a, x_b], dim=1)
        log_det_jac = -log_scale.sum(dim=1)
        return x, log_det_jac