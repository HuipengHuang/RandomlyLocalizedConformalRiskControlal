import argparse
import os
import imageio
import numpy as np
import shutil
from my_utils import jpg2png
from segmentation.PraNet.lib.PraNet_Res2Net import PraNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils.dataloader import get_loader, PolypDataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument("--batch_size", type=int, default=16)
#parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet-ori.pth')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')
args = parser.parse_args()

#if False:
 #   jpg2png("./data/TestDataset/HyperKvasir/images")
  #  jpg2png("./data/TestDataset/HyperKvasir/masks")

#for _data_name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
for _data_name in ['HyperKvasir']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet/{}/'.format(_data_name)
    model = PraNet()
    model.load_state_dict(torch.load(args.pth_path))
    model.to("cuda")
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    #test_loader = test_dataset(image_root, gt_root, args.testsize)
    test_dataset = PolypDataset(image_root, gt_root, args.testsize)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    #for image, gt, name in test_loader:
    for idx, (image, gt, origin_image_path) in enumerate(test_loader):
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
        res = (res / 10).sigmoid().data

        res_np = (res * 255).cpu().numpy().astype(np.uint8)

        # Process ground truth (convert to 0-255 uint8)
        gt_np = (gt* 255).cpu().numpy().astype(np.uint8)  # Remove channel dim

        # Save all files
        for i in range(bsz):
            name = str(idx * bsz+ i)
            print(f'> {_data_name} - {name}')
            imageio.imwrite(os.path.join(save_path, f'{name}_pred.png'), res_np[i].reshape((352, 352)))  # Prediction
            imageio.imwrite(os.path.join(save_path, f'{name}_gt.png'), gt_np[i].reshape((352, 352)))  # Ground truth
            shutil.copy2(origin_image_path[0], os.path.join(save_path, f'{name}_origin.jpg'))  # Original