from .thr import THR
from .raps import RAPS
from .saps import SAPS
from .aps import APS
import torch

class WeightedScore():
    def __init__(self, ):
        super().__init__()
        self.thr_score_function = THR()
        self.raps_score_function = RAPS(random=False, weight=1, size_regularization=10)
        self.saps_score_function = SAPS(random=False, weight=1)
        self.aps_score_function = APS(random=False)

    def __call__(self, weight, prob):
        thr_score = self.thr_score_function(prob)
        raps_score = self.raps_score_function(prob)
        saps_score = self.saps_score_function(prob)
        #aps_score = self.aps_score_function(prob)
        score = thr_score * weight[:, 0].view(-1, 1) + raps_score * weight[:, 1].view(-1, 1) + saps_score * weight[:, 2].view(-1, 1)
        #score += aps_score * weight[:, 3].view(-1, 1)
        return score

    def compute_target_score(self, weight, prob, target):
        """It will return a tensor with dimension 1"""
        score = self(weight, prob)
        return score[torch.arange(target.size(0)), target]