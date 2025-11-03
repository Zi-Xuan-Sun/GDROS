import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        if len(img2.shape) == 2:
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class opt_sar_train(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='./datasets/os_dataset', dstype='training'):
        super(opt_sar_train, self).__init__(aug_params)
        """OS dataset(png) whu dataset(tif)"""
        flow_root = osp.join(root, dstype, 'truth_flow')
        image_opt_root = osp.join(root, dstype, 'opt')
        image_sar_root = osp.join(root, dstype, 'sar_warped')

        if split == 'test':
            self.is_test = True

        image_optlist = sorted(glob(osp.join(image_opt_root, '*.png')))  # tif png
        image_sarlist = sorted(glob(osp.join(image_sar_root, '*.png')))

        for i in range(0, int(len(image_optlist))):
            self.image_list += [[image_optlist[i], image_sarlist[i]]]
            self.extra_info += [(i)]  # scene and frame_id

        if split != 'test':
            self.flow_list += sorted(glob(osp.join(flow_root, '*.flo')))



def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'OSdata':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_dataset = 100 * opt_sar_train(aug_params, split='training', dstype=args.dstype)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=0, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

class FlowDataset_test(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        if len(img2.shape) == 2:
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return self.image_list[index][0], img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class opt_sar_test(FlowDataset_test):
    def __init__(self, aug_params=None, split='testing', root='./datasets/os_dataset', dstype='testing'):

        super(opt_sar_test, self).__init__(aug_params)
        """OS dataset(png) whu dataset(tif)"""
        flow_root = osp.join(root, dstype, 'truth_flow')
        image_opt_root = osp.join(root, dstype, 'opt')
        image_sar_root = osp.join(root, dstype, 'sar_warped')

        if split == 'test':
            self.is_test = True

        image_optlist = sorted(glob(osp.join(image_opt_root, '*.png')))  # tif png
        image_sarlist = sorted(glob(osp.join(image_sar_root, '*.png')))
        print('total data number:', len(image_optlist))

        for i in range(0, int(len(image_optlist))):
            self.image_list += [[image_optlist[i], image_sarlist[i]]]
            self.extra_info += [(i)]  # scene and frame_id

        if split != 'test':
            self.flow_list += sorted(glob(osp.join(flow_root, '*.flo')))