import torch
from .base_score import BaseScore


class THR(BaseScore):
    def __call__(self, prob):
        return 1 - prob

    def compute_target_score(self, prob, target):
        """It will return a tensor with dimension 1."""
        target_prob = prob[torch.arange(len(target)), target]
        return 1 - target_prob
