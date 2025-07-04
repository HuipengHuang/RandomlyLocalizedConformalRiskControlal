import torch
import torch.nn as nn
from tqdm import tqdm
from PraNet.my_utils import get_fnr_list
class RandomlyLocalizedConformalRiskControl(nn.Module):
    def __init__(self, args, model, kernel_function):
        super(RandomlyLocalizedConformalRiskControl, self).__init__()
        self.args = args
        self.model = model
        self.alpha = args.alpha
        self.kernel_function = kernel_function

    def forward(self, cal_loader, test_loader):
        cal_res = torch.tensor([], device='cuda')
        cal_gt = torch.tensor([], device="cuda")

        for idx, (image, gt, origin_image_path) in tqdm(enumerate(cal_loader), desc="Compute calibration feature"):
            image = image.cuda()
            gt = gt.cuda()
            res = self.get_feature(image)
            cal_res = torch.cat((cal_res, res), dim=0)
            cal_gt = torch.cat((cal_gt, gt), dim=0)

        fdr_tensor = torch.tensor([], device="cuda")
        size = 0
        test_res = torch.tensor([], device="cuda")
        test_gt = torch.tensor([], device="cuda")
        for idx, (image, gt, origin_image_path) in tqdm(enumerate(test_loader), desc="Testing"):
            image = image.cuda()
            gt = gt.cuda()

            res = self.get_feature(image)
            bsz, c, h, w = gt.shape

            test_res = torch.cat((test_res, res), dim=0)
            test_gt = torch.cat((test_gt, gt), dim=0)

        p = self.kernel_function.get_p(cal_res.view(cal_res.shape[0], -1), test_res.view(test_res.shape[0], -1))

        for j in tqdm(range(test_gt.shape[0]), desc="Cauculate FDR for each test data"):
            _lambda = self.get_lambda(cal_res, cal_gt, p=p[j])
            pred = (test_res[j] >= 1 - _lambda).to(torch.int)

            size += torch.sum(pred).item()
            fdr = get_fnr_list(pred, test_gt[j])

            fdr_tensor = torch.cat((fdr_tensor, fdr), dim=0)

        result_dict = {"MeanFDR": torch.mean(fdr_tensor).item(),
                       "Var": torch.var(fdr_tensor).item(),
                       "Avg_size": size / fdr_tensor.shape[0]}

        return result_dict, fdr_tensor


    def get_feature(self, image):
        # Forward pass
        res5, res4, res3, res2 = self.model(image)
        res = res2
        # Get dimensions
        bsz, c, h, w = image.shape

        upsample = nn.Upsample(size=(h, w), mode='bilinear')
        res = upsample(res)
        res = (res / self.args.T).sigmoid().data

        return res

    def get_lambda(self, cal_res, cal_gt, p, B=1):
        low = 0
        high = 1
        tol = 1e-8
        _lambda = None

        for _ in range(1000):
            _lambda = (low + high) / 2

            pred = (cal_res >= 1 - _lambda).to(torch.int)
            R_n_list = get_fnr_list(pred, cal_gt)
            risk = p[:-1] @ R_n_list + (p[-1] * B)

            if risk == self.alpha:
                break
            elif risk < self.alpha:
                high = _lambda
            else:
                low = _lambda

            if high - low < tol:
                break
        _lambda = low
        return _lambda