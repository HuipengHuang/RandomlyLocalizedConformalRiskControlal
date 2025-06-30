import argparse
import os
import imageio
import numpy as np
import shutil
from my_utils import crc, rlcrc, plot_histgram
from lib.PraNet_Res2Net import PraNet
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.nn as nn
import torch
from utils.dataloader import get_loader, PolypDataset
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Goes up to local_crc
sys.path.append(str(project_root))

from kernel_function.utils import get_kernel_function

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cal_ratio", type=float, default=0.5, help="Calibration ratio")
#parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet-ori.pth')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')
parser.add_argument("--num_run", type=int, default=1, help="Number of runs")
parser.add_argument("--kernel_function", type=str, default="naive", help="Kernel function")
parser.add_argument("--h", type=float, default=1, help="hyperparameter for gaussian kernel")
parser.add_argument("--alpha", type=float, default=0.1, help="Risk")
parser.add_argument("--plot", default="False", choices=["True", "False"])
parser.add_argument("--output_dir", default="./plot_results/")
parser.add_argument("--T", default=1.0, type=float)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--dataset", default="all", choices=['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', "HyperKvasir", "Kvasirfamily", "polypgen_positive"])
args = parser.parse_args()

if args.dataset == "all":
    ds_name_list = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', "HyperKvasir", "polypgen_positive"]
elif args.dataset == "Kvasirfamily":
    ds_name_list = ["Kvasir", "HyperKvasir"]
else:
    ds_name_list = [args.dataset]
test_ds_list = []
for _data_name in ds_name_list:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet/{}/'.format(_data_name)

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    #test_loader = test_dataset(image_root, gt_root, args.testsize)
    test_ds_list.append(PolypDataset(image_root, gt_root, args.testsize))

model = PraNet()
model.load_state_dict(torch.load(args.pth_path))
model.to("cuda")
cal_test_dataset = ConcatDataset(test_ds_list)
cal_length = int(len(cal_test_dataset) * args.cal_ratio)
result_dict_list = []
all_fdr_tensor = torch.tensor([], device="cuda")
for i in range(args.num_run):
    cal_dataset, test_dataset = random_split(cal_test_dataset,[cal_length, len(cal_test_dataset) - cal_length])
    cal_loader = DataLoader(dataset=cal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    B = 1

    with torch.no_grad():
        model.eval()

        cal_res = torch.tensor([], device='cuda')
        cal_gt = torch.tensor([], device="cuda")

        for idx, (image, gt, origin_image_path) in tqdm(enumerate(cal_loader), desc="Compute calibration feature"):
            image = image.cuda()
            gt = gt.cuda()

            # Forward pass
            res5, res4, res3, res2 = model(image)
            res = res2
            # Get dimensions
            bsz, c, h, w = gt.shape

            # Process prediction
            upsample = nn.Upsample(size=(h, w), mode='bilinear')
            res = upsample(res)
            res = (res / args.T).sigmoid()

            #min_values = torch.min(res, dim=-1, keepdim=True).values
            #max_values = torch.max(res, dim=-1, keepdim=True).values
            #res = (res - min_values) / (max_values - min_values + 1e-6)

            cal_res = torch.cat((cal_res, res), dim=0)
            cal_gt = torch.cat((cal_gt, gt), dim=0)
        if args.kernel_function == "naive":
            result_dict, fdr_tensor = crc(cal_res, cal_gt, model, test_loader, alpha=args.alpha, args=args)
        else:
            kernel_function = get_kernel_function(args)
            result_dict, fdr_tensor = rlcrc(cal_res, cal_gt, model, test_loader, kernel_function=kernel_function, args=args)
        result_dict_list.append(result_dict)
        all_fdr_tensor = torch.cat((all_fdr_tensor, fdr_tensor), dim=0)
for key, value in result_dict_list[0].items():
    mean_value = np.mean([result_dict_list[j][key] for j in range(len(result_dict_list))])
    print(f"{key}: {value}")

if args.plot == "True":
    plot_histgram(all_fdr_tensor, args.alpha, args)