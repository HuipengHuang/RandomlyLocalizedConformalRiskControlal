import torch
from .base_score import BaseScore


class SAPS(BaseScore):
    """Paper: Conformal Prediction for Deep Classifier via Label Ranking
       Link: https://arxiv.org/pdf/2310.06430"""
    def __init__(self, random, weight):
        super().__init__()
        self.random = random
        self.weight = weight

    def __call__(self, prob):
        ordered_prob, indices = torch.sort(prob, descending=True)
        max_prob = ordered_prob[:, 0]

        if self.random:
            # Generate randomness.
            u = torch.rand(size=ordered_prob.shape,device=ordered_prob.device)
        else:
            u = torch.ones(size=ordered_prob.shape,device=ordered_prob.device)
        saps_score = torch.zeros_like(prob)

        saps_score[:,0] = max_prob * u[:, 0]

        for i in range(1, ordered_prob.shape[-1]):
            saps_score[:, i] = max_prob + (i - 1 + u[:, i]) * self.weight

        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        saps_score = saps_score.gather(dim=-1, index=sorted_indices)
        return saps_score

    def compute_target_score(self, prob, target):
        """It will return a tensor with dimension 1"""
        saps_score = self(prob)
        return saps_score[torch.arange(target.size(0)), target]
