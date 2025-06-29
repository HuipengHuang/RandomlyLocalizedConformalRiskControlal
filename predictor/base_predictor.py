import numpy as np

import torch
import torch.nn as nn
import math


class BasePredictor():
    def __init__(self, args, net, adapter_net=None):
        self.threshold = None
        self.alpha = args.alpha
        self.device = torch.device('cuda')
        self.args = args
        self.net = net
        self.B = None
    def calibrate(self, cal_loader, q=0, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.net.eval()



    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        if self.threshold is None:
            raise ValueError("Threshold score is None. Please do calibration first.")
        self.combined_net.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_prediction_set_size = 0
            class_coverage = [0 for i in range(100)]
            class_size = [0 for i in range(100)]
            total_samples = 0

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.shape[0]
                total_samples += batch_size

                logit = self.combined_net(data)
                prob = torch.softmax(logit, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                total_accuracy += (prediction == target).sum().item()

                batch_score = self.score_function(prob)
                prediction_set = (batch_score <= self.threshold).to(torch.int) * (
                            batch_score >= self.lower_threshold).to(torch.int)
                target_prediction_set = prediction_set[torch.arange(batch_size), target]
                total_coverage += target_prediction_set.sum().item()
                total_prediction_set_size += prediction_set.sum().item()
                for i in range(prediction_set.shape[0]):
                    class_coverage[target[i]] += 1
                    class_size[target[i]] += 1


            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_prediction_set_size / total_samples
            class_coverage_gap = np.array(class_coverage) / np.array(class_size)
            class_coverage_gap = np.sum(np.abs(class_coverage_gap - (1 - self.alpha))) / 100
            result_dict = {
                f"{self.args.score}_Top1Accuracy": accuracy,
                f"{self.args.score}_AverageSetSize": avg_set_size,
                f"{self.args.score}_Coverage": coverage,
                f"{self.args.score}_class_coverage_gap": class_coverage_gap,
            }
            return result_dict

