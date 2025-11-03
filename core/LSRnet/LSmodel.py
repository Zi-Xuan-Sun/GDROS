import torch
import torch.nn.functional as F
import cmath as c
import numpy as np
from torch import nn



class LSNet(nn.Module):
    ''' least squares regression'''
    def __init__(self, Npoint, num_classes):
        super(LSNet, self).__init__()
        self.num_classes = num_classes
        self.class_weights = nn.Embedding(num_classes, 1)
        self.rho = nn.Parameter(torch.ones([Npoint])) # 权重
        # self.N = F.normalize(D, p=2, dim=1)
        # 初始化权重参数
        self.weights = nn.Parameter(torch.randn(num_classes))
        # self.affine_params = nn.Parameter(torch.randn(2, 3), requires_grad=True)# 仿射矩阵的初始估计

    def one_hot(self, labels, num_classes):
        """
        Convert labels to one-hot encoding.
        :param labels: Tensor of shape [batch_size, height, width]
        :param num_classes: Number of classes
        :return: Tensor of shape [batch_size, num_classes, height, width]
        """
        batch_size, Npoints = labels.size()
        one_hot_labels = torch.zeros((batch_size, num_classes, Npoints), device=labels.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        return one_hot_labels

    def forward(self, X, F_, lbl):
        '''
        :param X: batch x Npoint x 3  Npoint个光学像素点构成的矩阵
        :param F: batch x Npoint x 2  Npoint个光流点构成的矩阵
        :param lbl: batch x Npoint x 1  Npoint个光流点构成的矩阵
        :return: trans_mat: batch x 3 x 2
        [a-1, e
         b, f-1
         c,   g]
        '''
        bs, N, _ = X.size()

        '''获取每个类别的权重'''
        # weights = self.class_weights(lbl).squeeze(-1)  # shape: (batch_size * Npoint, 1)

        # weights = self.N(weights)
        # weights = torch.sigmoid(weights)  # shape: (batch_size * Npoint, 1)
        '''不使用语义分类'''
        # weights = self.rho
        '''不使用加权'''
        # weights = torch.ones([N]).cuda()

        '''使用softmax函数处理权重参数'''
        lbl_one_hot = self.one_hot(lbl, self.num_classes)
        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(1, -1, 1)

        weights = weights * lbl_one_hot
        weights = torch.sum(weights, dim=1)
        # 将权重扩展到与 X 和 F 相同的形状
        weights = weights.view(bs, N, 1)

        weighted_X = X * weights
        weighted_F = F_ * weights
        # inv(X_t * W_t * W * X)
        tmp_mat = torch.linalg.inv(torch.bmm(torch.transpose(weighted_X, 1, 2), weighted_X))
        # X_t * W_t * W * F
        tmp_mat2 = torch.bmm(torch.transpose(weighted_X, 1, 2), weighted_F)

        trans_mat = torch.bmm(tmp_mat, tmp_mat2)
        trans_mat = torch.transpose(trans_mat, 1, 2) #转置
        # 使用学习到的参数进行微调
        # trans_mat = trans_mat + self.affine_params

        return trans_mat


def get_grid(batch_size, H, W, start=0):

    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()

    # grid = grid.to('cuda:1')
    grid[:, :2, :, :] = grid[:, :2, :, :] + start  # add the coordinate of left top
    return grid


def LSR_list(flow_predictions):
    trans_mat_predictions = []
    for flow in flow_predictions:
        bs, _, H, W = flow.size()
        N = H * W

        grid = get_grid(bs, H, W).permute(0, 2, 3, 1).float().reshape(bs, N, 3)
        X = grid  # [bs, N, 3]

        F_ = flow.permute(0, 2, 3, 1).reshape(bs, N, 2)  # [bs, N, 2]

        device = flow.device
        weights = torch.ones([bs, N], device=device).view(bs, N, 1)

        weighted_X = X * weights
        weighted_F = F_ * weights

        XtW = torch.transpose(weighted_X, 1, 2)  # [bs, 3, N]
        tmp_mat = torch.linalg.inv(torch.bmm(XtW, weighted_X))  # [bs, 3, 3]
        tmp_mat2 = torch.bmm(XtW, weighted_F)  # [bs, 3, 2]

        trans_mat = torch.bmm(tmp_mat, tmp_mat2).transpose(1, 2)  # [bs, 2, 3]
        trans_mat_predictions.append(trans_mat)

    return trans_mat_predictions

def LSR(bs, h, w, flow_pre1, Npoint):
    grid__ = get_grid(1, h, w).permute(0, 2, 3, 1)
    H_flow_f = grid__[:, :, :, :2] * 0
    grid = grid__.reshape(bs, -1, 3)
    input_flow_cat = flow_pre1.permute(0, 2, 3, 1).reshape(bs, -1, 2)[:, 10000: (10000 + Npoint)]
    grid_cat = grid[:, 10000: (10000 + Npoint), :]

    trans_mat = LSR_(grid_cat, input_flow_cat)
    for k in range(bs):
        pre_H = trans_mat[k, :, :]
        pre_H[0, 0] += 1
        pre_H[1, 1] += 1
        row_to_add = torch.tensor([[0, 0, 1]]).cuda()
        pre_H1 = torch.cat((pre_H, row_to_add), dim=0)
        pre_H1 = pre_H1.unsqueeze(0).repeat(h * w, 1, 1).unsqueeze(0)
        grid_ = grid__[0, :, :, :].reshape(-1, 3).unsqueeze(0).unsqueeze(3).float()
        grid_warp = torch.matmul(pre_H1, grid_)
        grid_warp = grid_warp.squeeze().reshape(h, w, 3).unsqueeze(0)
        flow_f = grid_warp[:, :, :, :2] / grid_warp[:, :, :, 2:] - grid__[:, :, :, :2]
        H_flow_f = torch.cat((H_flow_f, flow_f), 0)
    H_flow_f = H_flow_f[1:, ...]
    flow_pre = H_flow_f.permute(0, 3, 1, 2).cpu()
    return flow_pre

def LSR_(X, F_):
    '''
    :param X: batch x Npoint x 3  Npoint个光学像素点构成的矩阵
    :param F: batch x Npoint x 2  Npoint个光流点构成的矩阵
    :param lbl: batch x Npoint x 1  Npoint个光流点构成的矩阵
    :return: trans_mat: batch x 3 x 2
    [a-1, e
     b, f-1
     c,   g]
    '''
    bs, N, _ = X.size()

    weighted_X = X
    weighted_F = F_
    # inv(X_t * W_t * W * X)
    tmp_mat = torch.linalg.inv(torch.bmm(torch.transpose(weighted_X, 1, 2), weighted_X))
    # X_t * W_t * W * F
    tmp_mat2 = torch.bmm(torch.transpose(weighted_X, 1, 2), weighted_F)

    trans_mat = torch.bmm(tmp_mat, tmp_mat2)
    trans_mat = torch.transpose(trans_mat, 1, 2)


    return trans_mat