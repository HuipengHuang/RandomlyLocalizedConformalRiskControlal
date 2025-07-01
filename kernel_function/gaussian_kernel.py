from .base_kernel_function import BaseKernelFunction
import torch


class GaussianKernel(BaseKernelFunction):
    """H(x,x′) = 1 / ((2π * h**2)**d/2) * exp (−∥x−x′∥**2 / (2*h**2))"""
    def __init__(self, args, holdout_feature=None, h=1):
        super().__init__(args, holdout_feature)
        self.h = h

    def get_weight(self, cal_feature, test_feature):
        """
            Args:
                cal_feature shape: [calibration_set_size, feature_dim], test_feature shape: [batch_size, feature_dim]
                """
        if self.PCA is not None:
            cal_feature, test_feature = self.fit_transform(cal_feature, test_feature)
        d = cal_feature.shape[1]

        sampled_features = self.sample(test_feature)

        test_distance = torch.sum(((test_feature - sampled_features) / d)**2, dim=-1)

        # cal_distance shape: [batch_size, calibration_set_size]
        #cal_distance = torch.sum(((cal_feature - sampled_features.unsqueeze(dim=1)) / d) ** 2, dim=-1)
        cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
        for i in range(test_feature.shape[0]):

            cal_distance[i] = torch.sum(((cal_feature - sampled_features[i]) / d)**2, dim=-1)

        l2 = torch.cat((cal_distance, test_distance.unsqueeze(dim=1)), dim=1)

        # weight shape: [batch_size, calibration_set_size+1]
        weight = torch.exp(-l2 / (2 * self.h**2))
        weight = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
        #weight = torch.zeros_like(weight)
        #weight += 1 / weight
        return weight

    def sample(self, test_feature):
        "Sample feature shape: [batch_size, feature_dim]"
        #sampled_features = test_feature.clone().detach().to("cuda") + self.h * torch.randn_like(test_feature)
        sampled_features = torch.zeros_like(test_feature)
        for i in range(sampled_features.shape[0]):
            sampled_features[i] = test_feature[i] + torch.randn_like(test_feature[i]) * self.h
        return sampled_features


