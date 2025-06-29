import argparse
import os
import imageio
import numpy as np
import shutil
from my_utils import crc
from segmentation.PraNet.lib.PraNet_Res2Net import PraNet
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.nn as nn
import torch
from utils.dataloader import get_loader, PolypDataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--cal_ratio", type=float, default=0.5, help="Calibration ratio")
#parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet-ori.pth')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')
parser.add_argument("--num_run", type=int, default=10, help="Number of runs")
args = parser.parse_args()

#if False:
 #   jpg2png("./data/TestDataset/HyperKvasir/images")
  #  jpg2png("./data/TestDataset/HyperKvasir/masks")
test_ds_list = []
for _data_name in ['CVC-300', 'CVC-ClinicDB']:
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

for i in range(args.num_run):
    cal_dataset, test_dataset = random_split(cal_test_dataset,[cal_length, len(cal_test_dataset) - cal_length])
    cal_loader = DataLoader(dataset=cal_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    B = 1

    with torch.no_grad():
        model.eval()

        cal_res = torch.tensor([], device='cuda')
        cal_gt = torch.tensor([], device="cuda")

        for idx, (image, gt, origin_image_path) in enumerate(cal_loader):
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
            res = (res).sigmoid()

            cal_res = torch.cat((cal_res, res), dim=0)
            cal_gt = torch.cat((cal_gt, gt), dim=0)

        result_dict = crc(cal_res, cal_gt, model, test_loader, alpha=0.1)
        result_dict_list.append(result_dict)
        print(f"FDR: {result_dict["Mean FDR"]}")
for key, value in result_dict_list[0].items():
    mean_value = np.mean([result_dict_list[j][key] for j in range(len(result_dict_list))])
    print(f"{key}: {value}")