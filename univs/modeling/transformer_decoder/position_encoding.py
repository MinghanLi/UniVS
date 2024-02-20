# Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        self.dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        dim_t_z = torch.arange((num_pos_feats * 2), dtype=torch.float32)
        self.dim_t_z = temperature ** (2 * torch.div(dim_t_z, 2, rounding_mode='trunc') / (num_pos_feats * 2))

    def forward(self, x, mask=None):
        # b, t, c, h, w
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        pos_x = x_embed[:, :, :, :, None] / self.dim_t.to(x.device)
        pos_y = y_embed[:, :, :, :, None] / self.dim_t.to(x.device)
        pos_z = z_embed[:, :, :, :, None] / self.dim_t_z.to(x.device)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w
        return pos

    def forward_with_size(self, size, device):
        t, h, w = size
        not_mask = torch.ones((t, h, w), dtype=torch.bool, device=device)
        z_embed = not_mask.cumsum(0, dtype=torch.float32)  # t, h, w
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[-1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        pos_x = x_embed[:, :, :, None] / self.dim_t.to(device)
        pos_y = y_embed[:, :, :, None] / self.dim_t.to(device)
        pos_z = z_embed[:, :, :, None] / self.dim_t_z.to(device)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = (torch.cat((pos_y, pos_x), dim=3) + pos_z).permute(0, 3, 1, 2)  # t, c, h, w
        return pos

    def forward_points_with_size(self, size, xy_embed_normalized):
        device = xy_embed_normalized.device
        eps = 1e-6

        t, h, w = size
        z_not_mask = torch.ones((t, 1), dtype=torch.bool, device=device)
        z_embed = z_not_mask.cumsum(0, dtype=torch.float32)  # t, 1
        z_embed_normalized = z_embed / (z_embed[-1:] + eps)

        x_embed_normalized, y_embed_normalized = torch.unbind(xy_embed_normalized, dim=-1)
        x_embed_normalized = x_embed_normalized[None]  # 1, n
        y_embed_normalized = y_embed_normalized[None]  # 1, n

        if self.normalize:
            z_embed = z_embed_normalized * self.scale
            y_embed = y_embed_normalized * self.scale
            x_embed = x_embed_normalized * self.scale
        else:
            z_embed = z_embed
            y_embed = (y_embed_normalized * h).floor()
            x_embed = (x_embed_normalized * w).floor()

        pos_x = x_embed[:, :, None] / self.dim_t.to(device)
        pos_y = y_embed[:, :, None] / self.dim_t.to(device)
        pos_z = z_embed[:, :, None] / self.dim_t_z.to(device)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=-1).flatten(-2)  # 1, n, c/2
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=-1).flatten(-2)  # 1, n, c/2
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=-1).flatten(-2)  # t, 1, c
        pos = (torch.cat((pos_y, pos_x), dim=-1) + pos_z)  # t, n, c
        return pos


class PositionEmbeddingSine3DArbitraryT(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on videos.
    The input should be fixed spatial resolution, such as 1024*1024; 
    and the interval length of two frames should be provied: dt=t2-t1.
    This improved position embedding can support to re-segment entities from arbitrary two frames, such as [t, t+10].

    """

    def __init__(self, num_pos_feats=64, num_max_frames=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.num_max_frames = num_max_frames
        self.temperature = temperature
        self.normalize = normalize
        assert normalize, "Must enable normalization!"
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        self.dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        dim_t_z = torch.arange((num_pos_feats * 2), dtype=torch.float32)
        self.dim_t_z = temperature ** (2 * torch.div(dim_t_z, 2, rounding_mode='trunc') / (num_pos_feats * 2))

    def forward(self, x, t_indices=None, mask=None):
        # x: b, t, c, h, w
        # t_indices: b, t
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        b, t, _, h, w = x.shape
        if t_indices is None:
            t_indices = torch.arange(t, device=x.device)[None, :].repeat(b, 1)
        z_embed = t_indices[:, :, None, None].repeat(1,1,h,w).to(x.device)  # b, t, h, w

        assert mask is None
        mask = torch.zeros((b, t, h, w), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / self.num_max_frames * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        pos_x = x_embed[:, :, :, :, None] / self.dim_t.to(x.device)
        pos_y = y_embed[:, :, :, :, None] / self.dim_t.to(x.device)
        pos_z = z_embed[:, :, :, :, None] / self.dim_t_z.to(x.device)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w
        return pos

    def forward_with_size(self, size, t_indices=None, device='cpu'):
        # t_indices: t
        t, h, w = size
        if t_indices is None:
            t_indices = torch.arange(t, device=device)
        z_embed = t_indices[:, None, None].repeat(1,h,w).to(device)  # t, h, w

        not_mask = torch.ones((t, h, w), dtype=torch.bool, device=device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / self.num_max_frames * self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        pos_x = x_embed[:, :, :, None] / self.dim_t.to(device)
        pos_y = y_embed[:, :, :, None] / self.dim_t.to(device)
        pos_z = z_embed[:, :, :, None] / self.dim_t_z.to(device)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = (torch.cat((pos_y, pos_x), dim=3) + pos_z).permute(0, 3, 1, 2)  # t, c, h, w
        return pos

    def forward_points_with_size(self, size, xy_embed_normalized, t_indices=None):
        # xy_embed_normalized: n, 2
        device = xy_embed_normalized.device
        t, h, w = size

        if t_indices is None:
            t_indices = torch.arange(t, device=device)
        if isinstance(t_indices, list):
            t_indices = torch.as_tensor(t_indices, device=device)
        elif isinstance(t_indices, float) or isinstance(t_indices, int):
            t_indices = torch.as_tensor(t_indices, device=device)
        else:
            assert isinstance(t_indices, torch.Tensor), 'unvalid data type'
        assert t_indices.nelement() == 1 or t_indices.nelement() == t, 'Unvalid length for frame indices'
        if t_indices.nelement() == 1:
            t_indices = t_indices.repeat(t)
        z_embed = t_indices[:, None].to(device)  # t, 1

        x_embed_normalized, y_embed_normalized = torch.unbind(xy_embed_normalized, dim=-1)
        x_embed_normalized = x_embed_normalized[None]  # 1, n
        y_embed_normalized = y_embed_normalized[None]  # 1, n
        z_embed_normalized = z_embed / self.num_max_frames # t, 1

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed_normalized * self.scale
            y_embed = y_embed_normalized * self.scale
            x_embed = x_embed_normalized * self.scale
        else:
            z_embed = (z_embed_normalized * self.num_max_frames).floor()
            y_embed = (y_embed_normalized * h).floor()
            x_embed = (x_embed_normalized * w).floor()

        pos_x = x_embed[:, :, None] / self.dim_t.to(device)
        pos_y = y_embed[:, :, None] / self.dim_t.to(device)
        pos_z = z_embed[:, :, None] / self.dim_t_z.to(device)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=-1).flatten(-2)  # 1, n, c/2
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=-1).flatten(-2)  # 1, n, c/2
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=-1).flatten(-2)  # t, 1, c
        pos = (torch.cat((pos_y, pos_x), dim=-1) + pos_z)  # t, n, c
        return pos

