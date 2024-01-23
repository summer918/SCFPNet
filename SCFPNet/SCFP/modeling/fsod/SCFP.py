import copy

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import random
import seaborn
from timm.models.layers import to_2tuple, trunc_normal_
# from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
import numpy as np
from torchvision.transforms import Resize
import cv2
import time
# from detectron2.structures import Boxes, Instances
import math
import random

def drop_path(x, y, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x, y
    keep_prob = 1 - drop_prob

    shape_x = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor_x = keep_prob + torch.rand(shape_x, dtype=x.dtype, device=x.device)
    random_tensor_x.floor_()  # binarize
    output_x = x.div(keep_prob) * random_tensor_x

    shape_y = (y.shape[0],) + (1,) * (y.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor_y = keep_prob + torch.rand(shape_y, dtype=y.dtype, device=y.device)
    random_tensor_y.floor_()  # binarize
    output_y = y.div(keep_prob) * random_tensor_y
    return output_x, output_y


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x[0], x[1], self.drop_prob, self.training)


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

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


import os

# 多个注意力 空间上连接
class Multi_attention(nn.Module):
    def __init__(self, i, dim, atten_drop=0.):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(14)
        self.dim = dim
        self.linear_1 = nn.Conv2d(dim, dim // 4, 3, padding=3 // 2, bias=False)
        self.linear_2 = nn.Conv2d(dim, dim // 4, 3, padding=3 // 2, bias=False)
        self.conv2 = nn.Conv2d(dim, dim // 4, 3, padding=3 // 2, bias=False)
        self.norm2 = nn.LayerNorm(dim // 4)

        self.conv_R = nn.Conv2d(dim // 4, dim, 3, padding=3 // 2, bias=False)
        self.norm_R = nn.LayerNorm(dim)

        self.scale_1 = (dim // 4) ** -0.5
        self.scale = dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.relu = nn.ReLU()

        self.proj_norm = nn.LayerNorm(dim)

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

    def cross_attention(self, x, y):
        B_x, N_x, C_x = x.shape
        q = x
        k, v = y, y
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.atten_drop(attn)
        # 必须加上.transpose(1, 2).reshape(B_x, N_x, C_x)
        x_ = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C_x)
        x = x_ + x
        x = self.proj_norm(x)

        return x

    def sparse_attention(self, q, k, v):
        # .transpose(-2, -1)
        atten = (q.transpose(-2, -1) @ k) * self.scale_1
        atten = self.relu(atten)

        atten = self.atten_drop(atten)
        atten = v @ atten
        return atten

    def deal_q_s(self, q, s, h, w):

        q_ = self.avgpool(q)
        s_ = self.avgpool(s)

        B_s = q.shape[0]
        # 对V进行处理
        v2 = self.conv2(s_).flatten(2).transpose(1, 2).contiguous()
        v2 = self.norm2(v2)
        # 对qk进行处理[B H*W C]
        q_r = self.linear_1(q_).flatten(2).transpose(1, 2).contiguous()
        s_r = self.linear_2(s_).flatten(2).transpose(1, 2).contiguous()
        # # 方差归一化
        q_r = q_r / (torch.linalg.norm(q_r, dim=-1, keepdim=True))  # 方差归一化，即除以各自的模
        k = s_r / (torch.linalg.norm(s_r, dim=-1, keepdim=True))
        # 稀疏注意力
        x2 = self.sparse_attention(q_r, k, v2)
        # 变为原来的通道维度
        x2 = x2.reshape(B_s, 14, 14, -1).permute(0, 3, 1, 2).contiguous()
        x2 = self.conv_R(x2).flatten(2).transpose(1, 2).contiguous()
        x2 = self.norm_R(x2)
        # 与q_连接
        x = x2
        # [B H*W C]
        q_ = q_.flatten(2).transpose(1, 2).contiguous()
        # 在空间维度进行concat
        x = torch.cat((q_, x), dim=1)
        # 交叉注意力 变为原来的维度
        q = q.flatten(2).transpose(1, 2).contiguous()
        # 变为原来的维度
        q = self.cross_attention(q, x)
        return q

    def forward(self, q, s):
        # q1 = q.flatten(2).transpose(1, 2).contiguous()
        # s1 = s.flatten(2).transpose(1, 2).contiguous()
        # atten = (q1.transpose(-2, -1) @ s1) * self.scale
        # atten = self.softmax(atten)
        # for i in range(1):
        #     seaborn.heatmap(atten[i][:20,:20].cpu().detach()*10)
        #     plt.savefig(f'对齐之前{i}.jpg')
        #     plt.clf()
        B_q, _, h_q, w_q = q.shape
        B_s, _, h_s, w_s = s.shape
        if B_q == 1:
            s_ = s.mean(0, True)
        elif B_s == 1:
            s_ = s.repeat(B_q, 1, 1, 1)
        q_r = self.deal_q_s(q, s_, h_q, w_q)

        if B_q == 1:
            q_ = q.repeat(B_s, 1, 1, 1)
        elif B_s == 1:
            q_ = q.mean(0, True)
        s_r = self.deal_q_s(s, q_, h_s, w_s)
        # q1 = q_r
        # s1 = s_r
        # atten = (q1.transpose(-2, -1) @ s1) * self.scale
        # atten = self.softmax(atten)
        # for i in range(1):
        #     seaborn.heatmap(atten[i][:20,:20].cpu().detach()*10)
        #     plt.savefig(f'对齐之后{i}.jpg')
        #     plt.clf()

        return q_r, s_r

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
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
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

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

    def forward(self, x, H_x, W_x, y, H_y, W_y):
        B_x, N_x, C_x = x.shape
        B_y, N_y, C_y = y.shape
        assert B_x == 1 or B_y == 1

        q_x = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q_y = self.q(y).reshape(B_y, N_y, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if not self.linear:
            if self.sr_ratio > 1:
                # x1 = x[:, :H_x * W_x, :]
                # x_tmp = x[:, H_x * W_x:, :]
                x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
                x_ = self.sr(x_).reshape(B_x, C_x, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                # x_ = torch.cat((x_, x_tmp), dim=1)
                kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1,
                                                                                                      4).contiguous()

                y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
                y_ = self.sr(y_).reshape(B_y, C_y, -1).permute(0, 2, 1)
                y_ = self.norm(y_)
                kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1,
                                                                                                      4).contiguous()
            else:
                kv_x = self.kv(x).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1,
                                                                                                     4).contiguous()
                kv_y = self.kv(y).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1,
                                                                                                     4).contiguous()
        else:

            x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x).contiguous()
            x_ = self.sr(self.pool(x_)).reshape(B_x, C_x, -1).permute(0, 2, 1).contiguous()

            x_ = self.norm(x_)
            x_ = self.act(x_)

            kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1,
                                                                                                  4).contiguous()

            y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y).contiguous()

            y_ = self.sr(self.pool(y_)).reshape(B_y, C_y, -1).permute(0, 2, 1).contiguous()
            y_ = self.norm(y_)
            y_ = self.act(y_)
            kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1,
                                                                                                  4).contiguous()

        k_cat_x, v_cat_x = kv_x[0], kv_x[1]
        k_cat_y, v_cat_y = kv_y[0], kv_y[1]
        

        attn_x = (q_x @ k_cat_x.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        attn_x = self.attn_drop(attn_x)

        x = (attn_x @ v_cat_x).transpose(1, 2).reshape(B_x, N_x, C_x)
        x = self.proj(x)
        x = self.proj_drop(x)

       
        attn_y = (q_y @ k_cat_y.transpose(-2, -1)) * self.scale
        attn_y = attn_y.softmax(dim=-1)
        attn_y = self.attn_drop(attn_y)

        y = (attn_y @ v_cat_y).transpose(1, 2).reshape(B_y, N_y, C_y)
        y = self.proj(y)
        y = self.proj_drop(y)

        return x, y


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
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
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H_x, W_x, y, H_y, W_y):
        outs = self.drop_path(self.attn(self.norm1(x), H_x, W_x, self.norm1(y), H_y, W_y))
        x = x + outs[0]
        y = y + outs[1]

        x_1 = self.norm2(x)
        y_1 = self.norm2(y)
 
        outs = self.drop_path((self.mlp(x_1, H_x, W_x), self.mlp(y_1, H_y, W_y)))
 
        x = x + outs[0]
        y = y + outs[1]

        return x, y


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


def make_stage(i,
               img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
               num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
               attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
               sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None):
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    cur = 0
    for idx_ in range(i):
        cur += depths[idx_]

    patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                    patch_size=7 if i == 0 else 3,
                                    stride=4 if i == 0 else 2,
                                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                    embed_dim=embed_dims[i])

    block = nn.ModuleList([Block(
        dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
        sr_ratio=sr_ratios[i], linear=linear)
        for j in range(depths[i])])
    if i == 3:
        attention_matrix = Multi_attention(i, embed_dims[i])
    else:
        attention_matrix = None

    
    norm = norm_layer(embed_dims[i])

    return patch_embed, block, norm ,attention_matrix


class PyramidVisionTransformerV2(Backbone):  # (nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None, only_train_norm=False,
                 train_branch_embed=True, frozen_stages=-1, multi_output=False):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
#         self.conv_one = nn.Conv2d(320, 1, 3, 1, 1)
#         self.softmax = nn.Softmax(dim=1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
         

        self.branch_embed_stage = 0
        for i in range(num_stages):
            patch_embed, block, norm,attention_matrix  = make_stage(i,
                                                                    img_size, patch_size, in_chans, num_classes,
                                                                    embed_dims,
                                                                    num_heads, mlp_ratios, qkv_bias, qk_scale,
                                                                    drop_rate,
                                                                    attn_drop_rate, drop_path_rate, norm_layer,
                                                                    depths,
                                                                    sr_ratios, num_stages, linear, pretrained)

            if i >= self.branch_embed_stage:
                branch_embed = nn.Embedding(2, embed_dims[i])
                setattr(self, f"branch_embed{i + 1}", branch_embed)

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.pool=nn.AdaptiveAvgPool2d(14)
        self.apply(self._init_weights)
        self.multi_output = multi_output
        self.embed_dims = embed_dims
        self.frozen_stages = frozen_stages
        self.only_train_norm = only_train_norm
        self.train_branch_embed = train_branch_embed

        self._freeze_stages()
        if not self.train_branch_embed:
            self._freeze_branch_embed()

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            m.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.Linear):
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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.multi_output:
            outputs = {"res2": x[0], "res3": x[1], "res4": x[2], "res5": x[3]}
            return outputs
        else:
            outputs = {"res4": x[-1]}
            return outputs
    def featuremap_2_heatmap(self,feature_map):
        assert isinstance(feature_map, torch.Tensor)

        # 1*256*200*256 # feat的维度要求，四维
        feature_map = feature_map.detach()

        # 1*256*200*256->1*200*256
        heatmap = feature_map[:, 0, :, :] * 0
        for c in range(feature_map.shape[1]):
            heatmap += feature_map[:, c, :, :]
        heatmap = heatmap.cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap
    def forward_features_with_two_branch(self, x, y):
#         print(x.shape,y.shape)
        outs = []
#         print("特征提取？？？")

        B_x = x.shape[0]
        B_y = y.shape[0]
        for i in range(self.num_stages):
            if i >= self.branch_embed_stage:
                branch_embed = getattr(self, f"branch_embed{i + 1}")
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H_x, W_x = patch_embed(x)
            y, H_y, W_y = patch_embed(y)
            if i < self.branch_embed_stage:

                for blk in block:
                    x, y = blk(x, H_x, W_x, y, H_y, W_y)
            else:

                x_branch_embed = torch.zeros(x.shape[:-1], dtype=torch.long).cuda()
                x = x + branch_embed(x_branch_embed)

                y_branch_embed = torch.ones(y.shape[:-1], dtype=torch.long).cuda()
                y = y + branch_embed(y_branch_embed)

                for blk in block:
                    x, y = blk(x, H_x, W_x, y, H_y, W_y)

            x = norm(x)
            y = norm(y)

            x = x.reshape(B_x, H_x, W_x, -1).permute(0, 3, 1, 2).contiguous()
            y = y.reshape(B_y, H_y, W_y, -1).permute(0, 3, 1, 2).contiguous()
            
            outs.append((x, y))
        return outs
         
#         if id==2:
#             y=y.mean(0,True)
#             x1=self.pool(x).flatten(2).contiguous()
#             y1=self.pool(y).flatten(2).contiguous()
# #         torch.Size([1, 1900, 320]) torch.Size([1, 400, 320])
#             print(x1.shape,y1.shape,x.shape)
#             x2=x1[0][101].cpu().detach()
#             y2=y1[0][101].cpu().detach()
#             sim=x2.dot(y2) / (np.linalg.norm(x2) * np.linalg.norm(y2))
#             print(sim)
#             sim=sim.softmax(dim=-1)
#             print(sim.shape)
#             print(sim[0][101])
#             x=x[0]  
#             for i in range(x.shape[0]):
# #                     print(x.shape,x[0].shape)
#                     re=x[i]
#                     featuremap=re.unsqueeze(dim=0).unsqueeze(dim=0)
#                 #         print(x.shape,img.shape)
#                     heatmap = self.featuremap_2_heatmap(featuremap)
#                         # 200*256->512*640
#                     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的
#                         # 大小调整为与原始图像相同
#                     heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
#                         # 512*640*3
#                     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原
#                         # 始图像
#                 #         print(heatmap.shape,img.shape)
#                     superimposed_img = heatmap * 0.7 + 0.3 * img  # 热力图强度因子，修改参数，
#                         # 得到合适的热力图
#                     cv2.imwrite( f'channel{i}热力图.jpg',superimposed_img)  # 将图像保存
#             exit(0)

         

    # 测试的时候用这个函数提取特征
    def forward_with_two_branch(self, x, y):

        outs = self.forward_features_with_two_branch(x, y)

        if self.multi_output:
            outputs = {"res2": outs[0], "res3": outs[1], "res4": outs[2], "res5": outs[3]}
            return outputs
        else:
            outputs = {"res4": outs[-1]}
            return outputs

    def output_shape(self):
        if self.multi_output:
            return {
                "res2": ShapeSpec(
                    channels=self.embed_dims[0], stride=4
                ),
                "res3": ShapeSpec(
                    channels=self.embed_dims[1], stride=8
                ),
                "res4": ShapeSpec(
                    channels=self.embed_dims[2], stride=16
                ),
                "res5": ShapeSpec(
                    channels=self.embed_dims[3], stride=32
                )
            }
        else:
            return {
                "res4": ShapeSpec(
                    channels=self.embed_dims[2], stride=16
                )
            }

    def _freeze_branch_embed(self):
        print("冻结branch embed")
        for i in range(self.num_stages):
            if i >= self.branch_embed_stage:
                branch_embed = getattr(self, f"branch_embed{i + 1}")
                branch_embed.eval()
                for param in branch_embed.parameters():
                    param.requires_grad = False

    def _freeze_stages(self):
        print("===============frozen at ", self.frozen_stages)
        if self.only_train_norm:
            print("Only train the normalization layers")
            for i in range(2, 5):
                print("===============freezing stage ", i - 1)
                patch_embed = getattr(self, f"patch_embed{i - 1}")
                block = getattr(self, f"block{i - 1}")
                norm = getattr(self, f"norm{i - 1}")

                patch_embed.eval()
                for name, param in patch_embed.named_parameters():
                    if 'norm' in name:
                        if i < self.frozen_stages + 1:
                            param.requires_grad = False
                        else:
                            pass
                    else:
                        param.requires_grad = False

                block.eval()
                for name, param in block.named_parameters():
                    if 'norm' in name:
                        if i < self.frozen_stages + 1:
                            param.requires_grad = False
                        else:
                            pass
                    else:
                        param.requires_grad = False

                norm.eval()
                for name, param in norm.named_parameters():
                    if i < self.frozen_stages + 1:
                        param.requires_grad = False
                    else:
                        pass
        else:
            print("冻结参数")
            for i in range(2, self.frozen_stages + 1):
                print("===============freezing stage ", i - 1)
                patch_embed = getattr(self, f"patch_embed{i - 1}")
                block = getattr(self, f"block{i - 1}")
                norm = getattr(self, f"norm{i - 1}")

                patch_embed.eval()
                for param in patch_embed.parameters():
                    param.requires_grad = False
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False
                norm.eval()
                for param in norm.parameters():
                    param.requires_grad = False


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


# @BACKBONE_REGISTRY.register()
class pvt_v2_b0(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b1(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b2(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b2_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b3(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b3_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b3_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b4(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b4_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b5(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b5_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b5_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True,
            pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth",
            num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'],
            train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'],
            multi_output=kwargs['multi_output'])


@BACKBONE_REGISTRY.register()
def build_FCT_backbone(cfg, input_shape):
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    if backbone_type == "pvt_v2_b2_li":
        return pvt_v2_b2_li(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                            train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                            frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b5":
        return pvt_v2_b5(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                         train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                         frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b4":
        return pvt_v2_b4(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                         train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                         frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b3":
        return pvt_v2_b3(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                         train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                         frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b2":
        return pvt_v2_b2(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                         train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                         frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b1":
        return pvt_v2_b1(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                         train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                         frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b0":
        return pvt_v2_b0(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM,
                         train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED,
                         frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    else:
        print("do not support backbone type ", backbone_type)
        return None
