import torch

MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss, metrics

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


def matrix2flow(trans_mat_predictions_list, flow_gt):
    """ 将变换矩阵列表转换为对应的光流场列表 """
    mat_flow_list = []

    # 遍历输入列表中的每个变换矩阵张量
    for trans_mat_preds in trans_mat_predictions_list:
        bs = trans_mat_preds.size(0)
        device = trans_mat_preds.device

        # 自动获取图像尺寸（假设为256x256）
        _, _, H, W = flow_gt.size()

        # 生成坐标网格（优化设备适配）
        grid__ = get_grid(1, H, W).permute(0, 2, 3, 1).to(device)
        H_flow_f = torch.zeros(1, H, W, 2, device=device)  # 初始化流场容器

        # 批量处理改进版
        for k in range(bs):
            # 构造3x3齐次变换矩阵
            pre_H = trans_mat_preds[k].clone()  # [2, 3]
            pre_H[0, 0] += 1  # 恢复a参数
            pre_H[1, 1] += 1  # 恢复f参数

            # 添加最后一行[0,0,1]
            row_to_add = torch.tensor([[0, 0, 1]], device=device)
            pre_H1 = torch.cat([pre_H, row_to_add], dim=0)  # [3, 3]

            # 构建变换矩阵张量（优化维度扩展）
            pre_H1 = pre_H1.unsqueeze(0).expand(H * W, -1, -1).unsqueeze(0)  # [1, HW, 3, 3]

            # 准备齐次坐标（优化内存布局）
            grid_ = grid__[0].reshape(-1, 3)  # [HW, 3]
            grid_ = grid_.unsqueeze(0).unsqueeze(-1).float()  # [1, HW, 3, 1]

            # 执行坐标变换（优化矩阵乘法）
            grid_warp = torch.matmul(pre_H1, grid_)  # [1, HW, 3, 1]
            grid_warp = grid_warp.squeeze(-1).reshape(1, H, W, 3)  # [1, H, W, 3]

            # 计算光流场（增加数值稳定性）
            eps = 1e-7
            flow_f = (grid_warp[..., :2] / (grid_warp[..., 2:] + eps)) - grid__[..., :2]

            # 累积结果（优化内存分配）
            H_flow_f = torch.cat([H_flow_f, flow_f], dim=0)

        # 最终形状调整（优化维度顺序）
        H_flow_f = H_flow_f[1:].permute(0, 3, 1, 2)  # [bs, 2, H, W]
        mat_flow_list.append(H_flow_f)

    return mat_flow_list