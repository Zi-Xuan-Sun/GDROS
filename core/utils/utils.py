from __future__ import print_function
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
from numpy import sin, cos, tan
import random
from torch import linalg
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头部
        header = f.read(4)  # 读取文件头部标识
        assert header == b'PIEH', "Invalid .flo file"  # 校验文件头部标识
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]  # 读取图像宽度
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]  # 读取图像高度

        # 读取光流数据
        data = np.frombuffer(f.read(), dtype=np.float32).reshape((height, width, 2))

    return data

def save_correspondences_img(img1, img2, corr1, corr2, pred_corr2, results_dir, img_name):
    """ Save pair of images with their correspondences into a single image. Used for report"""
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1]), np.uint8)
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1.copy()
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2.copy()
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

    cv2.polylines(new_img, np.int32([corr1]), 1, (255, 0, 0), 2, cv2.LINE_AA)

    corr2_ = (corr2 + np.array([img1.shape[1], 0])).astype(np.int32)
    pred_corr2_ = (pred_corr2 + np.array([img1.shape[1], 0])).astype(np.int32)

    cv2.polylines(new_img, np.int32([corr2_]), 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.polylines(new_img, np.int32([pred_corr2_]), 1, (0, 225, 0), 2, cv2.LINE_AA)

    # Save image
    visual_file_name = os.path.join(results_dir, img_name)
    # cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    cv2.imwrite(visual_file_name, new_img)
    print('Wrote file %s' % visual_file_name)

def save_correspondences_img_scale2(img1, img1_sar, img2, corr1, corr2, pts1_wrap_scale1_inv, pred_corr2, pred_pts_scale1_inv, pred_pts2, results_dir, img_name):
    """ Save pair of images with their correspondences into a single image. Used for report"""
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img1_sar.shape[1]+img2.shape[1]), np.uint8)
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1.copy()
    new_img[0:img1_sar.shape[0], img1.shape[1]:img1.shape[1] + img1_sar.shape[1]] = img1_sar.copy()
    new_img[0:img2.shape[0], img1.shape[1] + img1_sar.shape[1]:img1.shape[1] + img1_sar.shape[1] + img2.shape[1]] = img2.copy()
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

    cv2.polylines(new_img, np.int32([corr1]), 1, (255, 0, 0), 2, cv2.LINE_AA)

    corr2_ = (corr2 + np.array([img1.shape[1], 0])).astype(np.int32)
    pred_corr2_ = (pred_corr2 + np.array([img1.shape[1], 0])).astype(np.int32)

    cv2.polylines(new_img, np.int32([corr2_]), 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.polylines(new_img, np.int32([pred_corr2_]), 1, (0, 225, 0), 2, cv2.LINE_AA)

    corr2_scale1 = (pts1_wrap_scale1_inv + np.array([img1.shape[1]+img1_sar.shape[1], 0])).astype(np.int32)
    pred_corr2_scale1 = (pred_pts_scale1_inv + np.array([img1.shape[1]+img1_sar.shape[1], 0])).astype(np.int32)
    pred_corr2_scale2 = (pred_pts2 + np.array([img1.shape[1] + img1_sar.shape[1], 0])).astype(np.int32)

    cv2.polylines(new_img, np.int32([corr2_scale1]), 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.polylines(new_img, np.int32([pred_corr2_scale1]), 1, (0, 225, 0), 2, cv2.LINE_AA)
    cv2.polylines(new_img, np.int32([pred_corr2_scale2]), 1, (0, 0, 225), 2, cv2.LINE_AA)

    # Save image
    visual_file_name = os.path.join(results_dir, img_name)
    # cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    cv2.imwrite(visual_file_name, new_img)
    print('Wrote file %s' % visual_file_name)

def inv_affine_matrix(A):
    # Calculate the inversed affine transformation matrix with 6 parameters.
    TA = A.size()
    B = torch.Tensor([[[0, 0, 1]]]).to(device)
    B = B.repeat(TA[0], 1, 1)
    A_ = torch.cat([A, B], dim=1)
    Inv = linalg.inv(A_)
    #Inv = torch.neg(A_, out=None)#szx 只平移
    Inv = Inv[:, 0:2, :]
    return Inv

def get_affine_matrix(img_size, range_):
    translation_pixel_x, translation_pixel_y, scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y = \
        get_all_random_parameters_from_range(range_)
    # Calculate the affine transformation matrix and its inverse matrix
    # from the given parameters of translation, scaling, rotation and shearing.
    # dx = translation_pixel_x * 2 / img_size
    # dy = translation_pixel_y * 2 / img_size
    dx = translation_pixel_x
    dy = translation_pixel_y
    sx = scale_x
    sy = scale_y
    theta = rotate_angle * np.pi / 180
    faix = shear_angle_x * np.pi / 180
    faiy = shear_angle_y * np.pi / 180
    A = torch.tensor(np.float32(np.array([[
        (sx * (cos(theta) - sin(theta) * tan(faiy)), sy * (cos(theta) * tan(faix) - sin(theta)), dx),
        (sx * (sin(theta) + cos(theta) * tan(faiy)), sy * (sin(theta) * tan(faix) + cos(theta)), dy)]]))).to(device)
    # A_inv = inv_affine_matrix(A)
    return A

def get_random_number_from_range(range_):
    if range_[2]:
        range_ = np.array(range_)
        t = 0
        while range_[0]%1 or range_[1]%1 or range_[2]%1: #判断dtype是否为整数
            range_ = range_ * 10
            t = t + 1
        range_ = np.array(range_, dtype=int)
        x_ = random.randrange(range_[0], range_[1], range_[2])
        if t:
            x_ = x_ * np.power(10., -t)
        return x_
    else:
        return range_[0]

def get_all_random_parameters_from_range(range_):
    range_translation_pixel_x = range_['range_translation_pixel_x']
    range_translation_pixel_y = range_['range_translation_pixel_y']
    range_scale_x = range_['range_scale_x']
    range_scale_y = range_['range_scale_y']
    range_rotate_angle = range_['range_rotate_angle']
    range_shear_angle_x = range_['range_shear_angle_x']
    range_shear_angle_y = range_['range_shear_angle_y']
    translation_x_equals_y = range_['translation_x_equals_y']
    scale_x_equals_y = range_['scale_x_equals_y']
    shear_x_equals_y = range_['shear_x_equals_y']
    translation_pixel_x = get_random_number_from_range(range_translation_pixel_x)
    translation_pixel_y = translation_pixel_x if translation_x_equals_y \
        else get_random_number_from_range(range_translation_pixel_y)
    scale_x = get_random_number_from_range(range_scale_x)
    scale_y = scale_x if scale_x_equals_y \
        else get_random_number_from_range(range_scale_y)
    rotate_angle = get_random_number_from_range(range_rotate_angle)
    shear_angle_x = get_random_number_from_range(range_shear_angle_x)
    shear_angle_y = shear_angle_x if shear_x_equals_y \
        else get_random_number_from_range(range_shear_angle_y)
    return translation_pixel_x, translation_pixel_y, scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y

def AffineTransform(tensor_, affine_matrix):
    TS = tensor_.size()
    TA = affine_matrix.size()
    if len(TS)==4:
        b = tensor_.size()[0]
    elif len(TS)==3 and TS[0]==1:
        b = 1
        tensor_ = tensor_.unsqueeze(0)
    elif len(TS)==2:
        b = 1
        tensor_ = tensor_.unsqueeze(0).unsqueeze(0)
    if len(TA)==2:
        affine_matrix = affine_matrix.repeat(b, 1, 1)
    t = tensor_.size()
    grid = F.affine_grid(affine_matrix, tensor_.size())
    tensor_warp = F.grid_sample(tensor_, grid)

    return tensor_warp

def AffineTransformFromRange(tensor_, range_):
    translation_pixel_x, translation_pixel_y, scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y = \
        get_all_random_parameters_from_range(range_)
    assert tensor_.size()[-1] == tensor_.size()[-2]
    img_size = tensor_.size()[-1]
    A, A_inv = get_affine_matrix(img_size, translation_pixel_x, translation_pixel_y,
                  scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y)
    tensor_warp = AffineTransform(tensor_, A)
    return tensor_warp, A, A_inv

def show_flow_arrow(flow_show, flow):
    '''image ndarray (256, 256, 3)
    flow ndarray (32, 32, 2)
    points ndarray (32*32, 2)'''

    # 获取x和y方向的位移
    flow_x = flow[0]
    flow_y = flow[1]

    # 创建网格点来均匀采样光流箭头
    grid_y, grid_x = np.mgrid[64:512:96, 64:512:96]  # 以16像素间隔采样点

    # 获取对应采样点的位移
    sampled_flow_x = flow_x[grid_y, grid_x]
    sampled_flow_y = flow_y[grid_y, grid_x]

    # 绘制彩色光流箭头图
    plt.figure(figsize=(5, 5))
    plt.imshow(flow_show / 255.0)  # 创建一个空白图像
    plt.quiver(grid_x, grid_y, sampled_flow_x, sampled_flow_y, color='r', angles='xy', scale_units='xy', scale=1.3,
               width=0.01, headwidth=3, headlength=4, pivot='tail', alpha=0.7)
    plt.axis('off')
    # plt.savefig('/home/sunzixuan/实验数据盘/test86_flow21_arrow.png')
    # plt.show()

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def indexing(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    """
        TODO: directly indexing features instead of sampling
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode='nearest')

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def drawMatches(image1, image2, point_map, inliers=None, max_points=1000):
    """
    inliers: set of (x1, y1) points
    """
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape

    matchImage = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
    matchImage[:rows1, :cols1, :] = np.dstack([image1] * 3)
    matchImage[:rows2, cols1:cols1 + cols2, :] = np.dstack([image2] * 3)

    small_point_map = [point_map[i] for i in np.random.choice(len(point_map), max_points)]

    # draw lines
    for x1, y1, x2, y2 in small_point_map:
        point1 = (int(x1), int(y1))
        point2 = (int(x2 + image1.shape[1]), int(y2))
        color = BLUE if inliers is None else (
            GREEN if (x1, y1, x2, y2) in inliers else RED)

        cv2.line(matchImage, point1, point2, color, 1)

    # Draw circles on top of the lines
    for x1, y1, x2, y2 in small_point_map:
        point1 = (int(x1), int(y1))
        point2 = (int(x2 + image1.shape[1]), int(y2))
        cv2.circle(matchImage, point1, 5, BLUE, 1)
        cv2.circle(matchImage, point2, 5, BLUE, 1)

    return matchImage

def showchessboard(img1, img2, save, path, name): #  ndarray(256,256)
    # # 缩放图像到256*256
    # img1 = img1.squeeze(0).permute(1,2,0).cpu().numpy()
    # img2 = img2.squeeze(0).permute(1,2,0).cpu().numpy()
    # 获取图片的大小
    h, w = img1.shape[:2]
    # 构建空白画布
    board = np.zeros((h, w), np.uint8)
    board_size = 64
    t = int(h/board_size)

    # 循环遍历画棋盘格
    for i in range(t):
        for j in range(t):
            if (i+j) % 2 == 0:
                # 第偶数个小方格显示img1的内容
                board[i*board_size:(i+1)*board_size, j*board_size:(j+1)*board_size] = img1[i*board_size:(i+1)*board_size, j*board_size:(j+1)*board_size]
            else:
                # 第奇数个小方格显示img2的内容
                board[i*board_size:(i+1)*board_size, j*board_size:(j+1)*board_size] = img2[i*board_size:(i+1)*board_size, j*board_size:(j+1)*board_size]
    if save is True:
        save_path = path
        save_name = name
        cv2.imwrite(save_path + '/' + save_name, board)

    # 显示拼接结果
    # cv2.imshow('board', board)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return board

def get_flow_from_h(h, height, width ):

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    transformed_grid = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=2)
    transformed_grid = np.matmul(transformed_grid, h.T)
    flow_H_pre = transformed_grid[:, :, :2] - np.stack([grid_x, grid_y], axis=2)
    flow_H_pre = torch.from_numpy(flow_H_pre).permute(2, 0, 1).float()

    return flow_H_pre