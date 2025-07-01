import torch
from .base_score import BaseScore


class APS(BaseScore):
    """Paper: Classification with Valid and Adaptive Coverage
       Link: https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf"""
    def __init__(self,random):
        super().__init__()
        self.random = random

    def __call__(self, prob):
        ordered_prob, indices = torch.sort(prob, descending=True)
        aps_score = torch.cumsum(ordered_prob, dim=-1)
        if self.random:
            #  Generate randomness
            aps_score -= torch.rand(size=aps_score.shape, device=aps_score.device) * ordered_prob

        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        aps_score = aps_score.gather(dim=-1, index=sorted_indices)
        return aps_score

    def compute_target_score(self, prob, target):
        """It will return a tensor with dimension 1"""
        aps_score = self(prob)
        return aps_score[torch.arange(target.size(0)), target]
