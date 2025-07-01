import torch
from .base_score import BaseScore


class RAPS(BaseScore):
    """Paper: Uncertainty Sets for Image Classifiers using Conformal Prediction
       Link: https://arxiv.org/pdf/2009.14193"""
    def __init__(self, random, weight, size_regularization):
        super().__init__()
        self.random = random
        self.weight = weight
        self.k_reg = size_regularization

    def __call__(self, prob):
        ordered_prob, indices = torch.sort(prob, descending=True)
        aps_score = torch.cumsum(ordered_prob, dim=-1)
        if self.random:
            aps_score -= torch.rand(size=aps_score.shape, device=aps_score.device) * ordered_prob

        regularization = torch.reshape(torch.arange(aps_score.shape[-1]) + 1 - self.k_reg, (1, -1))

        raps_score = aps_score + self.weight * torch.maximum(regularization.to(prob.device), torch.zeros_like(regularization, device=prob.device))

        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)

        raps_score = raps_score.gather(dim=-1, index=sorted_indices)
        return raps_score

    def compute_target_score(self, prob, target):
        """It will return a tensor with dimension 1"""
        saps_score = self(prob)
        return saps_score[torch.arange(target.size(0)), target]
