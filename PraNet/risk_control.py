#Example CUDA_VISIBLE_DEVICES=1 python risk_control.py --cal_ratio 0.8 --num_runs 10 --kernel_function gaussian --h 1 --batch_size 32 --dataset HyperKvasir --plot True
import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent  # Adjust if needed
sys.path.append(str(project_root))

import argparse
import numpy as np
from my_utils import plot_histogram, get_dataset
from lib.PraNet_Res2Net import PraNet
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch
import torch.nn as nn
from risk_control.rlcrc import RandomlyLocalizedConformalRiskControl
from risk_control.crc import ConformalRiskControl
from kernel_function.utils import get_kernel_function

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cal_ratio", type=float, default=0.5, help="Calibration ratio")
#parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet-ori.pth')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')
parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
parser.add_argument("--kernel_function", type=str, default="naive", help="Kernel function")
parser.add_argument("--h", type=float, default=None, help="hyperparameter for gaussian kernel")
parser.add_argument("--alpha", type=float, default=0.1, help="Risk")
parser.add_argument("--plot", default="False", choices=["True", "False"])
parser.add_argument("--output_dir", default="./plot_results/")
parser.add_argument("--T", default=1.0, type=float)
parser.add_argument("--num_workers", default=4, type=int)
#parser.add_argument("--dataset", default="all", choices=['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', "HyperKvasir", "Kvasirfamily", "polypgen_positive"])
parser.add_argument(
    "--dataset",
    nargs='+',  # Accepts 1 or more values
    default=["all"],  # Default as list
    choices=['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB',
             'Kvasir', "HyperKvasir", "Kvasirfamily", "polypgen_positive", "all"],
    help="Specify one or more datasets"
)

parser.add_argument("--holdout_dataset", default=None)
parser.add_argument("--pca", default=None, choices=["pca", "ppca", "kernel_pca", "sparse_pca", "ppca", "robust_pca"])
parser.add_argument("--n_components", type=int, default=2)
parser.add_argument("--efficient", default="True", help="Hyperparameter for PCA")
parser.add_argument("--efficient_calibration_size", default=None, type=int)
args = parser.parse_args()


with torch.no_grad():
    model = PraNet()
    model.eval()
    model.load_state_dict(torch.load(args.pth_path))
    model.to("cuda")
    cal_test_dataset, holdout_dataset = get_dataset(args)
    cal_length = int(len(cal_test_dataset) * args.cal_ratio)
    args.calibration_size = cal_length
    holdout_feature = None
    if holdout_dataset is not None:
        holdout_dataloader = DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)
        holdout_feature = torch.tensor([], device="cuda")
        for image, gt, _ in holdout_dataloader:
            image, gt = image.to("cuda"), gt.to("cuda")
            res5, res4, res3, res2 = model(image)
            res = res2
            # Get dimensions
            bsz, c, h, w = image.shape

            upsample = nn.Upsample(size=(h, w), mode='bilinear')
            res = upsample(res)
            res = (res / args.T).sigmoid().data
            holdout_feature = torch.cat((holdout_feature, res), dim=0)
        holdout_feature = holdout_feature.view(holdout_feature.shape[0], -1)

    if args.kernel_function != "naive":
        kernel_function = get_kernel_function(args, holdout_feature)
        crc = RandomlyLocalizedConformalRiskControl(args, model, kernel_function)
    else:
        crc = ConformalRiskControl(args, model)

    result_dict_list = []
    all_fdr_tensor = torch.tensor([], device="cuda")

    for i in range(args.num_runs):
        cal_dataset, test_dataset = random_split(cal_test_dataset,[cal_length, len(cal_test_dataset) - cal_length])
        cal_loader = DataLoader(dataset=cal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        B = 1

        result_dict, fdr_tensor = crc(cal_loader, test_loader)

        result_dict_list.append(result_dict)
        all_fdr_tensor = torch.cat((all_fdr_tensor, fdr_tensor), dim=0)
    for key, value in result_dict_list[0].items():
        mean_value = np.mean([result_dict_list[j][key] for j in range(len(result_dict_list))])
        print(f"{key}: {value}")

    if args.plot == "True":
        plot_histogram(all_fdr_tensor, args.alpha, args)