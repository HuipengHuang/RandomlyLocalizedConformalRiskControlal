import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent  # Adjust if needed
sys.path.append(str(project_root))
import argparse
import numpy as np
from models.utils import build_model
from dataset.utils import build_dataloader
from kernel_function.utils import get_kernel_function
import torch
from predictors.local_predictor import RandomlyLocalizedPredictor
from predictors.predictor import Predictor
from predictors.utils import plot_histogram

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="resnet50", help='Choose neural network architecture.')
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet"],
                    help="Choose dataset for training.")
parser.add_argument("--pretrained", default="True")
parser.add_argument("--load", type=str, default=None, help="Load pretrained weights.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--num_runs", default=1, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--plot", default="True", choices=["True", "False"])
parser.add_argument("--output_dir", default="./plot_results/")
#  Hyperpatameters for Conformal Prediction
parser.add_argument("--alpha", type=float, default=0.1, help="Error Rate")
parser.add_argument("--score", type=str, default="thr", choices=["thr", "aps", "raps", "saps", "weight_score"])
parser.add_argument("--cal_ratio", type=float, default=0.5,
                    help="Ratio of calibration data's size. (1 - cal_ratio) means ratio of test data's size")
parser.add_argument("--kernel_function", type=str, default="naive", help="Kernel function")
#  Hyperparamters for PCA
parser.add_argument("--pca", default=None, choices=["pca", "ppca", "kernel_pca", "sparse_pca", "ppca", "robust_pca"])
parser.add_argument("--n_components", type=int, default=2)
parser.add_argument("--efficient", default="True", help="PCA Hyperparamter")

parser.add_argument("--vae", default=None, choices=["vae", "svae"])
parser.add_argument("--latent_dim", type=int, default=None)

parser.add_argument("--h", type=float, default=None)
parser.add_argument("--efficient_calibration_size", default=None, type=int)
args = parser.parse_args()

num_classes = 1000
if args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "cifar100":
    num_classes = 100

net = build_model(args, num_classes=num_classes)
mean_result_dict = {
                f"Top1Accuracy": 0,
                f"AverageSetSize": 0,
                f"Coverage": 0,
                f"WorstClassCoverage": 0,
                f"class_coverage_gap": 0.0
            }


net.eval()
class_coverage_numpy = np.zeros(shape=(num_classes,))
class_size_numpy = np.zeros(shape=(num_classes,))

for _ in range(args.num_runs):
    holdout_dataloader, cal_dataloader, test_dataloader, num_classes = build_dataloader(args)
    holdout_feature = torch.tensor([], device="cuda")
    with torch.no_grad():
        for data, target in holdout_dataloader:
            data, target = data.cuda(), target.cuda()
            feature = net.get_feature(data)
            holdout_feature = torch.cat((holdout_feature, feature))

    if args.kernel_function == "naive":
        predictor = Predictor(args, net)
    else:
        kernel_function = get_kernel_function(args, holdout_feature)
        predictor = RandomlyLocalizedPredictor(args, net, kernel_function=kernel_function)

    result_dict, class_coverage, class_size = predictor.evaluate(cal_dataloader, test_dataloader)
    class_size_numpy = class_size_numpy + class_size
    print(type(class_coverage_numpy), type(class_coverage), class_coverage_numpy.shape, class_coverage.shape)
    class_coverage_numpy = class_coverage_numpy + class_coverage * (class_size + 1e-6)

    for key, value in result_dict.items():
        print(key, value)
        mean_result_dict[key] += value
    print()

print("Mean Result")
for key, value in mean_result_dict.items():
    print(key, value / args.num_runs)

if args.plot == "True":
    plot_histogram(class_coverage_numpy/class_size_numpy, args.alpha, args)