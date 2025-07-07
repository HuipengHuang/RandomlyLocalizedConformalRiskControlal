import numpy as np
from classification.scores.utils import get_score
import torch
import math
from tqdm import tqdm

class Predictor:
    def __init__(self, args, net):
        self.score_function = get_score(args)
        self.net = net
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
        self.args = args

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.net.eval()
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_score = torch.tensor([], device=self.device)
            for data, target in cal_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.net(data)

                prob = torch.softmax(logits, dim=1)
                batch_score = self.score_function.compute_target_score(prob, target)

                cal_score = torch.cat((cal_score, batch_score), 0)
            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            self.threshold = threshold
            return threshold

    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, cal_loader, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        threshold = self.calibrate(cal_loader)
        num_classes = self.args.num_classes
        set_size_coverage = torch.zeros(size=(self.args.num_classes+1,), device="cuda")
        set_size_num = torch.zeros(size=(self.args.num_classes+1,), device="cuda")

        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_prediction_set_size = 0
            class_coverage = [0 for i in range(num_classes)]
            class_size = [0 for i in range(num_classes)]
            total_samples = 0

            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.shape[0]
                total_samples += batch_size

                logit = self.net(data)
                prob = torch.softmax(logit, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                total_accuracy += (prediction == target).sum().item()

                batch_score = self.score_function(prob)
                prediction_set = (batch_score <= threshold).to(torch.int)


                target_prediction_set = prediction_set[torch.arange(batch_size), target]
                total_coverage += target_prediction_set.sum().item()

                total_prediction_set_size += prediction_set.sum().item()

                for i in range(prediction_set.shape[0]):
                    class_coverage[target[i]] += prediction_set[i, target[i]].item()
                    class_size[target[i]] += 1
                    set_size_coverage[torch.sum(prediction_set[i])] += prediction_set[i, target[i]].item()
                    set_size_num[torch.sum(prediction_set[i])] += 1


            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_prediction_set_size / total_samples
            class_coverage = np.array(class_coverage) / (np.array(class_size) + 1e-6)

            if self.args.dataset == "imagenet":
                sscv_list = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
                sscv_list[0] += torch.sum(set_size_coverage[:2]).item() / (torch.sum(set_size_num[:2]).item()+1e-6)

                sscv_list[1] += torch.sum(set_size_coverage[2:4]).item() / (torch.sum(set_size_num[2:4]).item() +1e-6)

                sscv_list[2] += torch.sum(set_size_coverage[4:7]).item() / (torch.sum(set_size_num[4:7]).item()+1e-6)

                sscv_list[3] += torch.sum(set_size_coverage[7:11]).item() / (torch.sum(set_size_num[7:11]).item()+1e-6)

                sscv_list[4] += torch.sum(set_size_coverage[11:100]).item() / (torch.sum(set_size_num[11:100]).item()+1e-6)

                sscv_list[5] += torch.sum(set_size_coverage[100:]).item() / (torch.sum(set_size_num[100:]).item()+1e-6)

                sscv_list = abs(sscv_list[sscv_list!=0] - (1 - self.args.alpha))
                sscv = torch.max(sscv_list).item()
            else:
                raise NotImplementedError

            class_coverage_gap = np.sum(np.abs(class_coverage - (1 - self.alpha))) / num_classes
            result_dict = {
                f"Top1Accuracy": accuracy,
                f"AverageSetSize": avg_set_size,
                f"Coverage": coverage,
                f"WorstClassCoverage": np.min(class_coverage),
                f"class_coverage_gap": class_coverage_gap,
                f"SSCV": sscv,
                f"SSCV_list": sscv_list
            }

        return result_dict, class_coverage, np.array(class_size)