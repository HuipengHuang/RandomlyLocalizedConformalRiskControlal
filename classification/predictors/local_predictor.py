import os.path

import numpy as np
from classification.scores.utils import get_score
import torch
from tqdm import tqdm

class RandomlyLocalizedPredictor:
    def __init__(self, args, model, kernel_function):
        self.score_function = get_score(args)
        self.kernel_function = kernel_function
        self.model = model
        self.threshold = None
        self.alpha = args.alpha
        self.device = "cuda"
        self.args = args

    def calibrate(self, cal_feature, test_feature):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""



    def evaluate(self, cal_loader, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""

        num_classes = self.args.num_classes
        with torch.no_grad():
            cal_feature = torch.tensor([], device="cuda")
            cal_prob = torch.tensor([], device="cuda")
            cal_target = torch.tensor([], device="cuda", dtype=torch.int)
            for data, target in cal_loader:
                data = data.to("cuda")
                target = target.to("cuda")
                feature = self.model.get_feature(data)
                logits = self.model.feature2logits(feature)
                cal_prob = torch.cat((cal_prob, logits), dim=0)

                cal_feature = torch.cat((cal_feature, feature), 0)
                cal_target = torch.cat((cal_target, target), 0)

            cal_prob = torch.softmax(cal_prob, dim=-1)
            cal_score = self.score_function.compute_target_score(cal_prob, cal_target)

            total_coverage = 0
            total_prediction_set_size = 0
            class_coverage = [0 for i in range(num_classes)]
            class_size = [0 for i in range(num_classes)]
            total_samples = 0

            test_feature = torch.tensor([], device="cuda")
            test_target = torch.tensor([], device="cuda", dtype=torch.int)
            test_prob = torch.tensor([], device="cuda")

            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.shape[0]
                total_samples += batch_size

                feature = self.model.get_feature(data)
                logits = self.model.feature2logits(feature)

                test_feature = torch.cat((test_feature, feature), 0)
                test_prob = torch.cat((test_prob, logits), dim=0)
                test_target = torch.cat((test_target, target), 0)

            test_prob = torch.softmax(test_prob, dim=-1)
            total_accuracy = torch.sum(torch.argmax(test_prob, dim=-1) == test_target)

            test_score = self.score_function(test_prob)

            self.plot_feature_distance(cal_feature, test_feature)
            p = self.kernel_function.get_p(cal_feature, test_feature)

            threshold = self.get_weighted_quantile(
                torch.cat((cal_score, torch.tensor([1.0], device="cuda")), dim=0), p, alpha=self.alpha)

            prediction_set = (test_score <= threshold.view(-1, 1)).to(torch.int)

            target_prediction_set = prediction_set[torch.arange(test_target.shape[0]), test_target]
            total_coverage += target_prediction_set.sum().item()

            total_prediction_set_size += prediction_set.sum().item()

            for i in range(prediction_set.shape[0]):
                class_coverage[test_target[i]] += prediction_set[i, test_target[i]].item()
                class_size[test_target[i]] += 1


            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_prediction_set_size / total_samples
            class_coverage = np.array(class_coverage) / (np.array(class_size) + 1e-6)
            class_coverage_gap = np.sum(np.abs(class_coverage - (1 - self.alpha))) / num_classes
            result_dict = {
                f"Top1Accuracy": accuracy,
                f"AverageSetSize": avg_set_size,
                f"Coverage": coverage,
                f"WorstClassCoverage": np.min(class_coverage),
                f"class_coverage_gap": class_coverage_gap,
            }

        return result_dict, class_coverage, np.array(class_size)

    def get_weighted_quantile(self, score, weight, alpha):
        """
        Args:
            score: Tensor of shape (cal_size+1,) containing scores
            weight: Tensor of shape (batch_size, cal_size+1) where each row sums to 1
            alpha: Quantile level (e.g., 0.1 for 90% quantile)
        Returns:
            quantiles: Tensor of shape (batch_size,) containing the (1-alpha) weighted quantile per row
        """
        batch_size = weight.shape[0]
        quantiles = torch.zeros(batch_size, device="cuda")

        for i in range(batch_size):
            # Combine scores and weights for the current row

            row_weights = weight[i]

            # Sort scores and corresponding weights
            sorted_scores, sorted_indices = torch.sort(score)
            sorted_weights = row_weights[sorted_indices]

            # Compute cumulative weights
            cum_weights = torch.cumsum(sorted_weights, dim=0)

            # Find the smallest score where cumulative weight >= (1 - alpha)
            mask = cum_weights >= (1 - alpha)
            if torch.any(mask):
                quantile_idx = torch.nonzero(mask, as_tuple=True)[0][0]
                quantiles[i] = sorted_scores[quantile_idx]
            else:
                quantiles[i] = sorted_scores[-1]  # Fallback to max score if no quantile found

        return quantiles

    def plot_feature_distance(self, cal_feature, test_feature):
        d = test_feature.shape[1]
        cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
        for i in range(test_feature.shape[0]):
            cal_distance[i] = torch.sum(((cal_feature - test_feature[i]) / d) ** 2, dim=-1)**(0.5)
        min_val = torch.min(cal_distance)
        max_val = torch.max(cal_distance)
        normalized_distance = (cal_distance - min_val) / (max_val - min_val + 1e-8)
        import matplotlib.pyplot as plt
        plt.hist(normalized_distance.view(-1).cpu().numpy(), bins=100)
        i = 0
        path = f"./plot_results/distance.pdf{i}"

        while os.path.exists(path):
            i += 1
            path = f"./plot_results/distance.pdf{i}"

        plt.savefig(path)
        plt.show()
