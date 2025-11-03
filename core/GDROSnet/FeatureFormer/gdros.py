import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import FeatureEncoder
from .cnn import BasicEncoder
from ...update import BasicUpdateBlock, SmallUpdateBlock
from ...corr import CorrBlock
from ...utils.utils import bilinear_sampler, coords_grid, upflow8


class GDROS(nn.Module):
    def __init__(self, cfg):
        super(GDROS, self).__init__()
        self.cfg = cfg
        self.featureNet = FeatureEncoder(cfg)
        # self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')

        if 'alternate_corr' not in self.cfg:
            self.cfg.alternate_corr = False
        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = cfg.corr_levels = 4
        self.corr_radius = cfg.corr_radius = 4

        self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=self.hidden_dim)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, output=None, flow_init=None, test_mode=False):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0  # tensor[bs, 3, 256, 256]

        data = {}
        hdim = self.hidden_dim
        cdim = self.context_dim

        context = self.context_encoder(image1)  # tensor[bs, 256, 32, 32]

        feat_s, feat_t = self.featureNet(image1, image2, data, context)  # tensor[bs*32*32, 8, 128]

        # flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
        fmap1 = feat_s.float()
        fmap2 = feat_t.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        """ run the context network """

        cnet = self.context_encoder(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume torch.Size([5, 324, 46, 46])

            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions
