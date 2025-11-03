import sys
sys.path.append('core')
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from configs.default import get_cfg
import core.datasets
from core.GDROSnet.FeatureFormer.gdros import GDROS
from core.LSRnet.LSmodel import LSR

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def Registration_index_calculation(epe_list, T):
    epe_values_less_than_T = [value for value in epe_list if value < T]
    MAE = np.mean(epe_values_less_than_T) if epe_values_less_than_T else 0
    RMSE = np.var(epe_values_less_than_T) if epe_values_less_than_T else 0
    CMR = 100 * (len(epe_values_less_than_T) / len(epe_list))
    return MAE, RMSE, CMR

@torch.no_grad()
def validate(model):
    model.eval()
    results = {}

    val_dataset = core.datasets.opt_sar_test(split='testing', dstype=args.dstype, root=args.path)

    epe_list = []
    MAE_list = []

    for val_id in range(len(val_dataset)):
        if val_id % 50 == 0:
            print(val_id)

        file_opt, image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        bs, _, h, w = image1.size()

        _, flow_pre1 = model(image1, image2, iters=32, test_mode=True)

        flow_pre = LSR(bs, h, w, flow_pre1, args.Npoint)

        epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        epe_ = np.mean(epe.view(-1).numpy())
        print("Validation (%s) LSR EPE: %f" % (args.dstype, epe_))
        MAE_list.append(epe_)

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (args.dstype, epe, px1, px3, px5))
    results[f"{args.dstype}_tile"] = np.mean(epe_list)


    MAE = np.mean(MAE_list)
    RMSE = np.var(MAE_list)
    MAE_1, RMSE_1, CMR_1 = Registration_index_calculation(MAE_list, 1)
    MAE_2, RMSE_2, CMR_2 = Registration_index_calculation(MAE_list, 2)
    MAE_3, RMSE_3, CMR_3 = Registration_index_calculation(MAE_list, 3)
    MAE_4, RMSE_4, CMR_4 = Registration_index_calculation(MAE_list, 4)
    MAE_5, RMSE_5, CMR_5 = Registration_index_calculation(MAE_list, 5)

    print("test (%s) EPE: %f, RMSE: %f" % (args.dstype, MAE, RMSE))
    print("T=1 CMR: %.2f%%, MAE: %.4f, RMSE: %.4f" % (CMR_1, MAE_1, RMSE_1))
    print("T=2 CMR: %.2f%%, MAE: %.4f, RMSE: %.4f" % (CMR_2, MAE_2, RMSE_2))
    print("T=3 CMR: %.2f%%, MAE: %.4f, RMSE: %.4f" % (CMR_3, MAE_3, RMSE_3))
    print("T=4 CMR: %.2f%%, MAE: %.4f, RMSE: %.4f" % (CMR_4, MAE_4, RMSE_4))
    print("T=5 CMR: %.2f%%, MAE: %.4f, RMSE: %.4f" % (CMR_5, MAE_5, RMSE_5))


    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./checkpoints/GDROS-os512.pth", help='load model')
    parser.add_argument('--mismatch_removal', default='LSR', help='--use LSR or not')
    parser.add_argument('--dstype', default="testing", type=str)
    parser.add_argument('--path', default='./datasets/os_dataset',
                        help="dataset for evaluation") # os数据集 whu512数据集
    parser.add_argument('--Npoint', type=int, nargs='+', default=200000)
    parser.add_argument('--num_classes', type=int, nargs='+', default=8)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--eval', default='featureflow_validation', help='eval benchmark')
    parser.add_argument('--small', action='store_true', help='use small model')
    args = parser.parse_args()

    exp_func = None
    cfg = None
    if args.eval == 'featureflow_validation':
        exp_func = validate
        cfg = get_cfg()
    else:
        print(f"EROOR: {args.eval} is not valid")
    cfg.update(vars(args))

    print(cfg)
    model = torch.nn.DataParallel(GDROS(cfg[cfg.transformer]))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()
    t1 = time.time()
    exp_func(model.module)
    t2 = time.time()
    print('total time:', (t2-t1))
