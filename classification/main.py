import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent  # Adjust if needed
sys.path.append(str(project_root))
import argparse
from model.utils import build_model
from dataset.utils import build_dataloader
from kernel_function.utils import get_kernel_function
import torch
from .predictor.local_predictor import RandomlyLocalizedPredictor
from .predictor.predictor import Predictor

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="resnet50", help='Choose neural network architecture.')
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet"],
                    help="Choose dataset for training.")
parser.add_argument("--load", type=str, default=None, help="Load pretrained weights.")
#  Hyperpatameters for Conformal Prediction
parser.add_argument("--alpha", type=float, default=0.1, help="Error Rate")
parser.add_argument("--score", type=str, default="thr", choices=["thr", "aps", "raps", "saps", "weight_score"])
parser.add_argument("--cal_ratio", type=float, default=0.5,
                    help="Ratio of calibration data's size. (1 - cal_ratio) means ratio of test data's size")
parser.add_argument("--kernel_function", type=str, default="naive", help="Kernel function")
parser.add_argument("--h", type=float, default=1)
parser.add_argument("--num_runs", default=1, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--pca", default=None, choices=["pca", "ppca", "kernel_pca", "sparse_pca", "ppca", "robust_pca"])
parser.add_argument("--n_components", type=int, default=2)

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
            }

with torch.no_grad():
    net.eval()
    for _ in range(args.num_runs):
        holdout_dataloader, cal_dataloader, test_dataloader, num_classes = build_dataloader(args)
        holdout_feature = torch.tensor([])
        for data, target in holdout_dataloader:
            data, target = data.cuda(), target.cuda()
            logits = net(data)
            holdout_feature = torch.cat((holdout_feature, logits))

        if args.kernel_function == "naive":
            predictor = Predictor(args, net)
        else:
            kernel_function = get_kernel_function(args, holdout_feature)
            predictor = RandomlyLocalizedPredictor(args, net, kernel_function=kernel_function)

        result_dict = predictor.evaluate(cal_dataloader, test_dataloader)
        for key, value in result_dict.items():
            print(key, value)
            mean_result_dict[key] += value

    print()
    print("Mean Result")
    for key, value in mean_result_dict.items():
        print(key, value / args.num_runs)
