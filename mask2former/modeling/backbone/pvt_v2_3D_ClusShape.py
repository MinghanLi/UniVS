import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from einops import rearrange, repeat

from detectron2.modeling import Backbone, BACKBONE_REGISTRY, ShapeSpec


def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        # x = token_dict['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                            (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        # start = time.time()
        # print(dist_matrix.shape, index_down.shape)
        dist_matrix = vector_gather(dist_matrix, index_down)
        # print('dist_matrix',dist_matrix.shape)
        # end = time.time()
        # total_time_dict['index_points1'] = total_time_dict['index_points1'] + end - start
        idx_cluster = dist_matrix.argmin(dim=1)
        # print('idx_cluster',idx_cluster.shape)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster


def merge_tokens(x, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    # x = token_dict['x']
    # idx_token = token_dict['idx_token']
    # agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6

    # print(all_weight.shape, token_weight.shape)
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    return x_merged


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, T, H, W):
        B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, T, H, W)
        x = x.reshape(B * T, H, W, C).permute(0,3,1,2)
        x = self.dwconv(x).reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, T, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, T, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False, n_frames=5, use_ctm=False, st_sa=False, cluster_num=128, shape_embed=None,
                 dynamic_3dpos=False, downsample=None, divide=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear
        self.sr_ratio = sr_ratio
        self.use_ctm = use_ctm
        self.st_sa = st_sa
        self.cluster_num = cluster_num
        self.dynamic_3dpos = dynamic_3dpos
        self.downsample = downsample
        self.divide = divide

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        
        if self.use_ctm:
            self.ctm = CTM(dim, n_frames, cluster_num=self.cluster_num, shape_embed=shape_embed, num_heads=num_heads)
            self.weight_ctm = nn.Parameter(torch.zeros(dim))
        if self.dynamic_3dpos:
            self.pos_conv = nn.Conv3d(self.dim, self.dim, 3, 1, 1, groups=self.dim)
        if self.downsample is not None:
            self.weight_downsample = nn.Parameter(torch.zeros(dim))
            if self.downsample == 'avgpool':
                self.downsample_op = nn.AdaptiveAvgPool3d((8, 2, 2))
            elif self.downsample == 'maxpool':
                self.downsample_op = nn.AdaptiveMaxPool3d((8, 2, 2))
            elif self.downsample == 'conv3d':
                self.downsample_op = nn.Conv3d(dim, dim, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 0, 0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, T, H, W):
        B, N, C = x.shape
        ## Spatial Attention only

        if self.dynamic_3dpos:
            x_pos = rearrange(x, 'B (T H W) C -> B C T H W', T=T, H=H, W=W)
            x_pos = rearrange(self.pos_conv(x_pos), 'B C T H W -> B (T H W) C')
            x = x_pos + x

        q = self.q(x)
        if self.use_ctm:
            q_ = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        elif self.downsample is not None:
            q_ = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.st_sa:
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BxnHxTHWxD
        else:
            q = q.reshape(B*T, N//T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BTxnHxHWxD

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.reshape(B*T, H, W, C).permute(0, 3, 1, 2)
                x_ = self.sr(x_).reshape(B*T, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                if self.st_sa:
                    x_ = x_.reshape(B, -1, C)
                if self.use_ctm:
                    # x_ = x_.reshape(B*T, -1, C)
                    x_d = self.ctm(x_.reshape(B, -1, C), T, H//self.sr_ratio, W//self.sr_ratio)
                    x_d = x_d.reshape(B, -1, C)
                    kv_ = self.kv(x_d).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                elif self.downsample is not None:
                    x_d = x_.reshape(B, -1, C).permute(0, 2, 1).reshape(B, C, T, H//self.sr_ratio, W//self.sr_ratio)
                    x_d = self.downsample_op(x_d).reshape(B, C, -1).permute(0, 2, 1)
                    kv_ = self.kv(x_d).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                elif self.divide is True:
                    short_cut = x_
                    x_ = x_.reshape(B, T, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4).reshape(-1, T, C // self.num_heads)
                    
                    attn_ = (x_ @ x_.transpose(-2, -1)) * self.scale
                    attn_ = attn_.softmax(dim=-1)
                    attn_ = self.attn_drop(attn_)
                    x_ = (attn_ @ x_).reshape(B,-1,self.num_heads,T, C//self.num_heads).permute(0, 3, 1, 2, 4).reshape(B*T, -1, C)
                    x_ = short_cut + x_

                if self.st_sa:
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x_).reshape(B*T, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                ## Spatial-temporal attention
                if self.use_ctm:
                    # x = x.reshape(B*T, -1, C)
                    # start = time.time()
                    x_d = self.ctm(x, T, H//self.sr_ratio, W//self.sr_ratio)
                    # end = time.time()
                    # total_time_dict['ctm'] = total_time_dict['ctm'] + end - start
                    x_d = x_d.reshape(B, -1, C)
                    kv_ = self.kv(x_d).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    if self.st_sa:
                        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
                    else:
                        kv = self.kv(x).reshape(B*T, -1, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
                else:
                    if self.downsample is not None:
                        x_d = x.reshape(B, -1, C).permute(0,2,1).reshape(B, C, T, H//self.sr_ratio, W//self.sr_ratio)
                        x_d = self.downsample_op(x_d).reshape(B,C,-1).permute(0,2,1)
                        kv_ = self.kv(x_d).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

                    if self.st_sa:
                        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    else:
                        kv = self.kv(x).reshape(B*T, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = rearrange(x, 'B (T H W) C -> (B T) C H W', T=T, H=H, W=W)
            x_ = self.sr(self.pool(x_)).flatten(-2).permute(0, 2, 1)  # BTxHW/pxC
            x_ = self.norm(x_)
            x_ = self.act(x_)

            if self.st_sa:
                # Spatial-temporal, 2xBxnHxTHW/pxD
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                # Spatial attention, 2xBTxnHxHW/pxD
                kv = self.kv(x_).reshape(B * T, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            if self.use_ctm:
                x_d = self.ctm(x_.reshape(B, -1, C), T, 7, 7)
                x_d = x_d.reshape(B, -1, C)
                # Spatial-temporal attention, 2xBxnHxTHW/pxD
                kv_ = self.kv(x_d).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # if not self.st_sa, q: BTxnHxHWxD, kv: BTxnHxHW/pxD --> BTxnHxHWxD
        # if self.st_sa,     q: BxnHxTHWxD, kv: BxnHxTHW/pxD --> BxnHxTHWxD
        # only spatial attention
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.use_ctm:
            # spatial temporal attention
            # q_: BxnHxTHWxD, kv_: BxnHxTHW/pxD --> BxnHxTHWxD, where actually q == q_
            k_, v_ = kv_[0], kv_[1]
            attn_ = (q_ @ k_.transpose(-2, -1)) * self.scale
            attn_ = attn_.softmax(dim=-1)
            attn_ = self.attn_drop(attn_)
            x_ = (attn_ @ v_).transpose(1, 2).reshape(B, N, C)
            x = x + x_ * self.weight_ctm
        elif self.downsample is not None:
            k_, v_ = kv_[0], kv_[1]
            attn_ = (q_ @ k_.transpose(-2, -1)) * self.scale
            attn_ = attn_.softmax(dim=-1)
            attn_ = self.attn_drop(attn_)
            x_ = (attn_ @ v_).transpose(1, 2).reshape(B, N, C)
            x = x + x_ * self.weight_downsample

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False,
                 use_ctm=False, st_sa=False, cluster_num=128, shape_embed=None,
                 dynamic_3dpos=False, downsample=None, divide=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            sr_ratio=sr_ratio, linear=linear, use_ctm=use_ctm, st_sa=st_sa,
            cluster_num=cluster_num, shape_embed=shape_embed, dynamic_3dpos=dynamic_3dpos,
            downsample=downsample, divide=divide)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_part1(self, x, T, H, W):
        return self.attn(self.norm1(x), T, H, W)
    
    def forward_part2(self, x, T, H, W):
        return self.drop_path(self.mlp(self.norm2(x), T, H, W))

    def forward(self, x, T, H, W):
        shortcut = x
        x = self.forward_part1(x, T, H, W)
        x = shortcut + self.drop_path(x)

        x = x + self.forward_part2(x, T, H, W)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        if patch_size == 7:
            patch_size = (3, 7, 7)
            stride = (1, 4, 4)
        else:
            patch_size = (1, patch_size, patch_size)
            stride = (1, 2, 2)
        
        assert max(patch_size) > max(stride), "Set larger patch_size than stride"

        if patch_size[0] == 2:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                  padding=(0, patch_size[1] // 2, patch_size[2] // 2))
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                  padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, T, H, W


class CTM(nn.Module):
    def __init__(self, embed_dim, n_frames=5, cluster_num=128, k=9, shape_embed=None, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.cluster_num = cluster_num
        self.score = nn.Linear(self.embed_dim, 1)
        self.k = k
        self.shape_embed = shape_embed
        self.num_heads = num_heads
        dim = 7*7*n_frames
        if self.shape_embed == 'conv':
            self.shape_conv = nn.Conv3d(1, embed_dim, 3, 1, 1)
            self.norm = nn.LayerNorm(embed_dim)
            self.act = nn.GELU()
        elif self.shape_embed == 'dynamic_3dpos':
            self.shape_conv = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        elif self.shape_embed == 'nonlinear':
            self.shape_conv = nn.Linear(dim, dim//2)
            self.act = nn.ReLU()
            self.w1_weight = nn.Parameter(torch.zeros(dim//2, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == 'nonlinear_lite':
            self.shape_conv = nn.Linear(dim, cluster_num)
            self.act = nn.GELU()
            self.w1_weight = nn.Parameter(torch.zeros(cluster_num, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == 'nonlinear_mh':
            self.shape_conv = nn.Linear(dim, dim//2)
            self.act = nn.ReLU()
            self.w1_weight = nn.Parameter(torch.zeros(dim//2, dim*num_heads))
            self.w1_bias = nn.Parameter(torch.ones(dim*num_heads))
            # self.norm = nn.LayerNorm(embed_dim)
            # self.act = nn.GELU()

        elif self.shape_embed == 'linear':
            # self.shape_conv = nn.Linear(dim, cluster_num)
            # self.act = nn.GELU()
            self.w1_weight = nn.Parameter(torch.zeros(dim, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == '3Dcoord_idx_embed':
            self.coord_conv = nn.Conv3d(4, embed_dim, 3, 1, 1)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(embed_dim)
        elif self.shape_embed == '3Dcoord_idx_nonlinear_shape_embed':
            self.coord_conv = nn.Conv3d(4, embed_dim, 3, 1, 1)
            self.act1 = nn.GELU()
            self.norm = nn.LayerNorm(embed_dim)
            self.shape_conv = nn.Linear(dim, dim//2)
            self.act2 = nn.ReLU()
            self.w1_weight = nn.Parameter(torch.zeros(dim//2, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == 'dynamic_3dpos_nonlinear':
            self.pos_conv = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
            self.shape_conv = nn.Linear(dim, dim//2)
            self.act = nn.ReLU()
            self.w1_weight = nn.Parameter(torch.zeros(dim//2, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == 'shape_mixer_lite':
            self.shape_conv = nn.Linear(dim, self.cluster_num)
            self.coord_conv = nn.Sequential(nn.Conv1d(4, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
            self.act = nn.GELU()
            self.w1_weight = nn.Parameter(torch.zeros(self.cluster_num, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == 'shape_mixer':
            self.shape_conv = nn.Linear(dim, dim//2)
            self.coord_conv = nn.Sequential(nn.Conv1d(4, 4 * 8, 1), nn.GELU(), nn.Conv1d(4*8, 1, 1))
            self.act = nn.GELU()
            self.w1_weight = nn.Parameter(torch.zeros(dim//2, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))
        elif self.shape_embed == 'dynamic_3dpos_shape_mixer_lite':
            self.pos_conv = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
            self.shape_conv = nn.Linear(dim, self.cluster_num)
            self.coord_conv = nn.Sequential(nn.Conv1d(4, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
            self.act = nn.GELU()
            self.w1_weight = nn.Parameter(torch.zeros(self.cluster_num, dim))
            self.w1_bias = nn.Parameter(torch.ones(dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, T, H, W):
        B, N, C = x.shape
        token_score = self.score(x)
        token_score = token_score.reshape(B, -1, 1)
        token_weight = token_score.exp()
        idx_cluster = cluster_dpc_knn(x, self.cluster_num, self.k)
        if self.shape_embed == 'conv':
            idx_cluster = idx_cluster.reshape(B, T, H, W, 1).permute(0, 4, 1, 2, 3)
            shape_embed = self.shape_conv(idx_cluster/self.cluster_num)
            shape_embed = self.act(self.norm(shape_embed.permute(0,2,3,4,1).reshape(B, -1, C)))
            x = x + shape_embed
        elif self.shape_embed == 'dynamic_3dpos':
            x_ = x.reshape(B, T, H, W, C).permute(0,4,1,2,3)
            x = self.shape_conv(x_).permute(0,2,3,4,1).reshape(B, N, C) + x
        elif self.shape_embed == 'linear':
            x_ = idx_cluster/self.cluster_num
            x_ = x_ @ self.w1_weight + self.w1_bias
            x = x * x_.unsqueeze(2).repeat(1,1,C)
        elif self.shape_embed == 'nonlinear' or self.shape_embed == 'nonlinear_lite':
            x_ = self.shape_conv(idx_cluster/self.cluster_num)
            x_ = self.act(x_)
            x_ = x_ @ self.w1_weight + self.w1_bias
            x = x * x_.unsqueeze(2).repeat(1,1,C)
        elif self.shape_embed == 'nonlinear_mh':
            x_ = self.shape_conv(idx_cluster/self.cluster_num)
            x_ = self.act(x_)
            x_ = x_ @ self.w1_weight + self.w1_bias
            x = x * x_.unsqueeze(2).repeat(1, 1, C//self.num_heads).reshape(B, N, C)
        elif self.shape_embed == '3Dcoord_idx_embed':
            device = x.device
            t_coord = torch.arange(0, T).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,H,W).unsqueeze(1) / T
            h_coord = torch.arange(0, H).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(B,T,1,W).unsqueeze(1) / H
            w_coord = torch.arange(0, W).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B,T,H,1).unsqueeze(1) / W
            idx_coord = idx_cluster.reshape(B, 1, T, H, W)/self.cluster_num
            coord = torch.cat((idx_coord, t_coord, h_coord, w_coord), dim=1)
            coord_embed = self.act(self.norm(self.coord_conv(coord).permute(0,2,3,4,1).reshape(B, N, C)))
            x = x + coord_embed
        elif self.shape_embed == '3Dcoord_idx_nonlinear_shape_embed':
            device = x.device
            # print(T, H, W)
            t_coord = torch.arange(0, T).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,H,W).unsqueeze(1) / T
            h_coord = torch.arange(0, H).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(B,T,1,W).unsqueeze(1) / H
            w_coord = torch.arange(0, W).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B,T,H,1).unsqueeze(1) / W
            idx_coord = idx_cluster.reshape(B, 1, T, H, W)/self.cluster_num
            coord = torch.cat((idx_coord, t_coord, h_coord, w_coord), dim=1)
            coord_embed = self.act1(self.norm(self.coord_conv(coord).permute(0,2,3,4,1).reshape(B, N, C)))
            x = x + coord_embed

            x_ = self.shape_conv(idx_cluster/self.cluster_num)
            x_ = self.act2(x_)
            x_ = x_ @ self.w1_weight + self.w1_bias

            x = x * x_.unsqueeze(2).repeat(1,1,C)
        elif self.shape_embed == 'dynamic_3dpos_nonlinear':
            x_ = x.reshape(B, T, H, W, C).permute(0,4,1,2,3)
            x = self.pos_conv(x_).permute(0,2,3,4,1).reshape(B, N, C) + x

            shape_feature = self.shape_conv(idx_cluster/self.cluster_num)
            shape_feature = self.act(shape_feature)
            shape_feature = shape_feature @ self.w1_weight + self.w1_bias
            x = x * shape_feature.unsqueeze(2).repeat(1,1,C)
        elif self.shape_embed == 'shape_mixer' or self.shape_embed == 'shape_mixer_lite':
            device = x.device
            t_coord = repeat(torch.arange(T).to(device), 'T ->B 1 T H W', B=B, H=H, W=W) / T
            h_coord = repeat(torch.arange(H).to(device), 'H ->B 1 T H W', B=B, T=T, W=W) / H
            w_coord = repeat(torch.arange(W).to(device), 'W ->B 1 T H W', B=B, T=T, H=H) / W
            idx_coord = idx_cluster.reshape(B, 1, T, H, W) / self.cluster_num
            coord = torch.cat((idx_coord, t_coord, h_coord, w_coord), dim=1)
            coord = coord.reshape(B, 4, N)
            x_ = self.coord_conv(coord).reshape(B, N) + idx_cluster/self.cluster_num
            x_ = self.shape_conv(x_)  # BxN -> Bx32
            x_ = self.act(x_)
            x_ = x_ @ self.w1_weight + self.w1_bias  # Bx32 -> BxN
            x = x * x_.unsqueeze(2).repeat(1, 1, C)

        elif self.shape_embed == 'dynamic_3dpos_shape_mixer' or self.shape_embed == 'dynamic_3dpos_shape_mixer_lite':
            x_pos = x.reshape(B, T, H, W, C).permute(0,4,1,2,3)
            x = self.pos_conv(x_pos).permute(0,2,3,4,1).reshape(B, N, C) + x
            device = x.device
            # print(T, H, W)
            t_coord = torch.arange(0,T).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,H,W).unsqueeze(1) / T
            h_coord = torch.arange(0,H).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(B,T,1,W).unsqueeze(1) / H
            w_coord = torch.arange(0,W).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B,T,H,1).unsqueeze(1) / W
            idx_coord = idx_cluster.reshape(B, 1, T, H, W)/self.cluster_num
            coord = torch.cat((idx_coord,t_coord,h_coord,w_coord),dim=1)
            coord = coord.reshape(B,4,N)
            x_ = self.coord_conv(coord).reshape(B,N) + idx_cluster/self.cluster_num
            x_ = self.shape_conv(x_)
            x_ = self.act(x_)
            x_ = x_ @ self.w1_weight + self.w1_bias
            x = x * x_.unsqueeze(2).repeat(1,1,C)

        down_dict = merge_tokens(x, idx_cluster, self.cluster_num, token_weight)

        return down_dict


class PVT_v2_3D_Clus_Shape(Backbone):
    def __init__(self, pretrained='/disk1/xiangwangmeng/pvt_v2_b2.pth', pretrained2d=True,
                 patch_size=4, in_chans=3, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[4, 2, 1, 1],
                 num_stages=4, out_features=None, n_frames=5, is_testing=False,
                 drop_path_rate=0.1, qk_scale=None, drop_rate=0., attn_drop_rate=0., linear=False,
                 use_ctm=[False, False, False, False], st_sa=False, cluster_num=128, shape_embed=None,
                 dynamic_3dpos=False, downsample=None, divide=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.fp16_enabled = True
        self.out_features = out_features
        self.n_frames = n_frames
        self.is_testing = is_testing
        self.depths = depths
        self.num_stages = num_stages
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.use_ctm = use_ctm

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self._out_feature_strides = {}
        self._out_feature_channels = {}
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear, use_ctm=self.use_ctm[i], st_sa=st_sa, cluster_num=cluster_num,
                shape_embed=shape_embed, dynamic_3dpos=dynamic_3dpos, downsample=downsample, divide=divide)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            stage = f'res{i + 2}'
            if stage in self.out_features:
                self._out_feature_channels[stage] = embed_dims[i]
                self._out_feature_strides[stage] = 4 * 2 ** i

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x_all):
        B = 1 if self.is_testing else x_all.shape[0] // self.n_frames
        x_all = rearrange(x_all, '(B T) C H W -> B C T H W', B=B)
        outs = {}

        T_all = x_all.shape[2]
        for t_start in range(0, T_all, 5):
            t_end = t_start + 5
            n_newly_frames = min(T_all, t_end) - t_start
            if t_end > T_all:
                t_start, t_end = T_all - 5, T_all

            x = x_all[:, :, t_start:t_end]
            if x.shape[2] < 5:
                x = x.repeat(1, 1, 5//x.shape[2] + 1, 1, 1)[:,:,-5:]

            for i in range(self.num_stages):
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                block = getattr(self, f"block{i + 1}")
                norm = getattr(self, f"norm{i + 1}")
                x, T, H, W = patch_embed(x)
                for blk in block:
                    x = blk(x, T, H, W)
                x = norm(x)
                x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

                name = f'res{i + 2}'
                if name in self.out_features:
                    if name not in outs.keys():
                        outs[name] = [x[:, :, -n_newly_frames:]]  # BxCxTxHxW
                    else:
                        outs[name].append(x[:, :, -n_newly_frames:])

        for k, v in outs.items():
            v = torch.cat(v, dim=2)
            assert v.shape[2] == T_all
            outs[k] = rearrange(v, 'B C T H W -> (B T) C H W')

        return outs

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        if self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def forward(self, x):
        """Forward function."""
        return self.forward_features(x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }


@BACKBONE_REGISTRY.register()
def build_pvtv2_b2_traj3d_backbone(cfg, input_shape):
    out_features = ["res3", "res4", "res5"]  # cfg.MODEL.PVTV2.OUT_FEATURES

    return PVT_v2_3D_Clus_Shape(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        linear=True,
        pretrained=cfg.MODEL.WEIGHTS,
        out_features=out_features,
        is_testing=cfg.MODEL.VITA.TESTING,
        n_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
        use_ctm=[True, True, True, True],
        cluster_num=32,
        shape_embed='shape_mixer_lite'
    )


@BACKBONE_REGISTRY.register()
def build_pvtv2_b2_base_backbone(cfg, input_shape):
    out_features = ["res3", "res4", "res5"]  # cfg.MODEL.PVTV2.OUT_FEATURES

    return PVT_v2_3D_Clus_Shape(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        linear=True,
        pretrained=cfg.MODEL.WEIGHTS,
        out_features=out_features,
        n_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
        is_testing=cfg.MODEL.VITA.TESTING
    )
