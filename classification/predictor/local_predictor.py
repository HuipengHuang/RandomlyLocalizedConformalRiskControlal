import numpy as np
from classification.scores.utils import get_score
import torch
from tqdm import tqdm

class RandomlyLocalizedPredictor:
    def __init__(self, args, net, kernel_function):
        self.score_function = get_score(args)
        self.kernel_function = kernel_function
        self.net = net
        self.threshold = None
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
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
            cal_target = torch.tensor([], device="cuda", dtype=torch.int)
            for data, target in cal_loader:
                data = data.to("cuda")
                target = target.to("cuda")
                logits = self.net(data)
                cal_feature = torch.cat((cal_feature, logits), 0)
                cal_target = torch.cat((cal_target, target), 0)
            cal_prob = torch.softmax(cal_feature, dim=-1)

            cal_score = self.score_function.compute_target_score(cal_prob, cal_target)

            total_accuracy = 0
            total_coverage = 0
            total_prediction_set_size = 0
            class_coverage = [0 for i in range(num_classes)]
            class_size = [0 for i in range(num_classes)]
            total_samples = 0

            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.shape[0]
                total_samples += batch_size

                logits = self.net(data)
                weight = self.kernel_function.get_weight(
                    cal_feature, logits)

                prob = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                total_accuracy += (prediction == target).sum().item()

                batch_score = self.score_function(prob)

                threshold = self.get_weighted_quantile(
                    torch.cat((cal_score, torch.tensor([1.0], device="cuda")), dim=0), weight, alpha=self.alpha)

                prediction_set = (batch_score <= threshold.view(-1, 1)).to(torch.int)

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
            class_coverage_gap = np.sum(np.abs(class_coverage_gap - (1 - self.alpha))) / num_classes
            result_dict = {
                f"Top1Accuracy": accuracy,
                f"AverageSetSize": avg_set_size,
                f"Coverage": coverage,
                f"WorstClassCoverage": np.min(class_coverage),
                #f"{self.args.score}_class_coverage_gap": class_coverage_gap,
            }

        return result_dict

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
        quantiles = torch.zeros(batch_size)

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