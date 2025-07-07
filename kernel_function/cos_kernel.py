from .base_kernel_function import BaseKernelFunction
import torch


class CosineKernel(BaseKernelFunction):
    """Cosine Kernel: K(x, x') = cos(θ) = (x·x') / (||x||·||x'||)"""

    def __init__(self, args, holdout_feature=None, holdout_target=None, h=None):
        super().__init__(args, holdout_feature, holdout_target, h)

    def calculate_weight(self, cal_feature, test_feature, sampled_features, h):
        """
        Compute cosine similarity weights between test/calibration features and sampled features.

        Args:
            cal_feature: Calibration features of shape [calibration_set_size, feature_dim]
            test_feature: Test features of shape [batch_size, feature_dim]
            sampled_features: Sampled features of shape [batch_size, feature_dim]
            h: Bandwidth (unused in cosine kernel but kept for API consistency)

        Returns:
            weights: Tensor of shape [batch_size, calibration_set_size + 1]
        """
        if h.dim() == 0:
            h = torch.zeros(size=(test_feature.shape[0], 1), device="cuda") + h

        # Normalize all features to unit vectors (cosine kernel is magnitude-invariant)
        cal_feature_norm = torch.nn.functional.normalize(cal_feature, p=2, dim=-1)
        test_feature_norm = torch.nn.functional.normalize(test_feature, p=2, dim=-1)
        sampled_features_norm = torch.nn.functional.normalize(sampled_features, p=2, dim=-1)

        # Compute cosine similarity: [batch_size, calibration_set_size]
        cal_sim = torch.matmul(sampled_features_norm, cal_feature_norm.T)  # [batch_size, cal_set_size]

        # Test similarity: [batch_size, 1]
        test_sim = torch.sum(test_feature_norm * sampled_features_norm, dim=-1, keepdim=True)

        # Concatenate similarities: [batch_size, cal_set_size + 1]
        similarities = torch.cat((cal_sim, test_sim), dim=-1)


        weight = torch.exp( - ((similarities - 1) / h) ** 2 / 2)  # h controls sharpness

        return weight

    def sample(self, test_feature, h):
        """
        Sample random perturbations around test features.
        For cosine kernel, we sample on the unit hypersphere.

        Args:
            test_feature: Input features of shape [batch_size, feature_dim]
            h: Controls perturbation magnitude (unused here but kept for API)

        Returns:
            sampled_features: Perturbed features on unit sphere [batch_size, feature_dim]
        """
        # Sample random unit vectors
        sampled_features = torch.zeros_like(test_feature)
        for i in range(test_feature.shape[0]):
            sampled_cos = 1 + h * torch.randn(1, device="cuda")
            while abs(sampled_cos) >= 1:
                sampled_cos = 1 + h * torch.randn(1, device="cuda")
            sampled_features[i] = sampled_cos
        return sampled_features