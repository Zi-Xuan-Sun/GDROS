from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.object = object
np.bool = bool
np.int = int
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from core.raft import RAFT
from torch.utils.data import DataLoader
import core.datasets as datasets
from core.loss import sequence_loss, matrix2flow
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger
from core.GDROSnet.FeatureFormer.gdros import GDROS
from core.LSRnet.LSmodel import LSR_list

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


# torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    model = nn.DataParallel(GDROS(cfg[cfg.transformer]))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model1 = torch.nn.DataParallel(RAFT(args))
        model1.load_state_dict(torch.load('./checkpoints/raft-sintel.pth'))
        model = model.module
        model1 = model1.module
        model.featureNet.feat_encoder.load_state_dict(model1.fnet.state_dict())
        model.context_encoder.load_state_dict(model1.fnet.state_dict())
        model.update_block.load_state_dict(model1.update_block.state_dict())
        model = nn.DataParallel(model)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)
    VAL_FREQ = 20000

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            output = {}
            flow_predictions = model(image1, image2, iters=cfg.iters)

            flow_loss, flow_metrics = sequence_loss(flow_predictions, flow, valid, cfg)

            """geometric loss"""
            trans_mat_predictions_list = LSR_list(flow_predictions)
            mat_flow_list = matrix2flow(trans_mat_predictions_list, flow)
            mat_flow_loss, mat_flow_metrics = sequence_loss(mat_flow_list, flow, valid, cfg)

            loss = flow_loss + mat_flow_loss
            metrics = {**flow_metrics, **mat_flow_metrics}


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)


            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()

    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)
    PATH1 = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH1)


    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gdros-os512', help="name your experiment")
    parser.add_argument('--dstype', default="training", type=str)
    parser.add_argument('--stage', default='OSdata', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    from configs.default import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
