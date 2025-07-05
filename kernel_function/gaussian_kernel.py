from .base_kernel_function import BaseKernelFunction
import torch


class GaussianKernel(BaseKernelFunction):
    """H(x,x′) = 1 / ((2π * h**2)**d/2) * exp (−∥x−x′∥**2 / (2*h**2))"""
    def __init__(self, args, holdout_feature=None, holdout_target=None, h=None):
        super().__init__(args, holdout_feature, holdout_target, h)

    def calculate_weight(self, cal_feature, test_feature, sampled_features, h):
        #  h could be a scaler, but also could be a tensor of shape [batch_size,]
        #  The first two line makes sure h is of shape [batch_size, ]
        if h.dim() == 0:
            h = torch.zeros(size=(test_feature.shape[0], 1), device="cuda") + h
        d = test_feature.shape[1]

        test_distance = torch.sum(((test_feature - sampled_features) / d / h) ** 2, dim=-1)

        # cal_distance shape: [batch_size, calibration_set_size]
        # cal_distance = torch.sum(((cal_feature - sampled_features.unsqueeze(dim=1)) / d) ** 2, dim=-1)
        cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
        for i in range(test_feature.shape[0]):
            cal_distance[i] = torch.sum(((cal_feature - sampled_features[i]) / d / h[i]) ** 2, dim=-1)

        l2 = torch.cat((cal_distance, test_distance.unsqueeze(dim=1)), dim=1)
        #  To avoid too large number in pytorch.
        l2 = l2 - torch.min(l2, dim=-1, keepdim=True)[0]

        # weight shape: [batch_size, calibration_set_size+1]
        weight = torch.exp(-l2 / 2)

        return weight

    def sample(self, test_feature, h):
        "Sample feature shape: [batch_size, feature_dim]"
        #sampled_features = test_feature.clone().detach().to("cuda") + self.h * torch.randn_like(test_feature)
        #for i in range(sampled_features.shape[0]):
        #    sampled_features[i] = test_feature[i] + torch.randn_like(test_feature[i]) * h * test_feature.shape[1]
        sampled_features = torch.randn_like(test_feature) * h * test_feature.shape[1] + test_feature
        return sampled_features


