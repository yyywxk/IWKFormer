#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/14 23:36
# @Author  : yyywxk
# @File    : model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from . import spatial_transform_local
from . import spatial_transform_local_feature
from .pconv import PartialConv2d
import matplotlib.pyplot as plt

# https://github.com/megvii-research/Portraits_Correction

# import spatial_transform_local

def shift2mesh(mesh_shift, width, height, grid_w, grid_h, cuda=False):
    '''

    :param mesh_shift: tensor
    :param width:
    :param height:
    :param grid_w:
    :param grid_h:
    :return:
    '''
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            # p = tf.constant([ww, hh], shape=[2], dtype=tf.float32)
            p = torch.Tensor([ww, hh])
            ori_pt.append(torch.unsqueeze(p, 0))
            # ori_pt.append(tf.expand_dims(p, 0))
    # ori_pt = tf.concat(ori_pt, axis=0)
    # ori_pt = tf.reshape(ori_pt, [grid_h + 1, grid_w + 1, 2])
    # ori_pt = tf.tile(tf.expand_dims(ori_pt, 0), [batch_size, 1, 1, 1])
    ori_pt = torch.cat(ori_pt, 0)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2).permute(2, 0, 1)
    # ori_pt = ori_pt.reshape(2, grid_h + 1, grid_w + 1)
    ori_pt = torch.unsqueeze(ori_pt, 0)
    ori_pt = ori_pt.repeat(batch_size, 1, 1, 1)
    if cuda:
        ori_pt = ori_pt.cuda()
    tar_pt = ori_pt + mesh_shift

    return tar_pt


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------PatchEmbed---------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(384, 512), patch_size=(4, 4), in_chans=3, embed_dim=48, norm_layer=nn.LayerNorm,
                 activation=F.gelu):
        """
        Image to Patch Embedding
        :param img_size:  Image size.  Default: 384 x 512.
        :param patch_size:  Patch token size. Default: [4, 4].
        :param in_chans:  Number of input image channels. Default: 3.
        :param embed_dim: Number of linear projection output channels. Default: 48.
        :param norm_layer: ormalization layer. Default: None
        """
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.activation = activation

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], "Input image size doesn't match model input requirement."
        x = self.layers(x)

        # N,C,PH,PW -> N,C,PH*PW -> N, PH*PW, C
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PartialPatchEmbed(nn.Module):
    def __init__(self, img_size=(384, 512), patch_size=(4, 4), in_chans=3, embed_dim=48, norm_layer=nn.LayerNorm,
                 activation=F.gelu):
        """
        Image to Patch Embedding
        :param img_size:  Image size.  Default: 384 x 512.
        :param patch_size:  Patch token size. Default: [4, 4].
        :param in_chans:  Number of input image channels. Default: 3.
        :param embed_dim: Number of linear projection output channels. Default: 48.
        :param norm_layer: ormalization layer. Default: None
        """
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.activation = activation

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.head = PartialConv2d(in_chans, embed_dim // 4, 3, padding=1)
        self.layers = nn.Sequential(
            # nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, mask):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], "Input image size doesn't match model input requirement."
        x = self.head(x, mask)
        x = self.layers(x)

        # N,C,PH,PW -> N,C,PH*PW -> N, PH*PW, C
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------WindowAttention---------------------------------
class PartialWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Window based multi-head self attention (W-MSA) module with relative position bias. It supports both of shifted
        and non-shifted window.
        :param dim: Number of input channels.
        :param window_size: The height and width of the window.
        :param num_heads: Number of attention heads.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: Override default qk scale of head_dim ** -0.5 if set
        :param attn_drop: Dropout ratio of attention weight. Default: 0.0
        :param proj_drop:  Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.computeq = nn.Linear(dim, dim, bias=qkv_bias)
        self.computekv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, partial_x, mask=None):
        """
        :param x: input features with shape of (num_windows*B, N, C)
        :param mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.computeq(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.computekv(partial_x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    :param x: (B, H, W, C)
    :param window_size: (int )window size
    :return: windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


# --------------------Window reverse---------------------------
def window_reverse(windows, window_size, H, W):
    """
    :param windows: (num_windows*B, window_size, window_size, C)
    :param window_size:  (int) Window size
    :param H: Height of image
    :param W: Width of image
    :return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# --------------------------------Swin Transformer Block------------------------
class PartialTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=F.gelu, norm_layer=nn.LayerNorm):
        """
        Introducion: Swin Transformer Block.
        :param dim: Number of input channels.
        :param input_resolution:  Input resulotion.
        :param num_heads: Number of attention heads.
        :param window_size: Window size.
        :param shift_size: Shift size for SW-MSA.
        :param mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        :param drop:  Dropout rate. Default: 0.0
        :param attn_drop: Attention dropout rate. Default: 0.0
        :param drop_path:  Stochastic depth rate. Default: 0.0
        :param act_layer:  Activation layer. Default: nn.GELU
        :param norm_layer:  Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) < self.window_size[0]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size[0] = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size[0], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = PartialWindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, partial_x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        partial_shortcut = partial_x
        partial_x = self.norm2(partial_x)
        partial_x = partial_x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            partial_shifted_x = torch.roll(partial_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            partial_shifted_x = partial_x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        partial_x_windows = window_partition(partial_shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        partial_x_windows = partial_x_windows.view(-1, self.window_size[0] * self.window_size[1],
                                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, partial_x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            partial_x = torch.roll(partial_shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            partial_x = partial_shifted_x
        x = x.view(B, H * W, C)
        partial_x = partial_x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x) + partial_x + partial_shortcut
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size[0]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size[0] = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size[0], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ---------------------------Patch Merging-----------------------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        """
        Patch Merging Layer
        :param input_resolution:   (tuple[int])Resolution of input feature.
        :param dim:  Number of input channels.
        :param norm_layer:  (nn.Module, optional) Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "x size are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# ---------------------------------------seqtoimg-------------------------------
def Seqtoimg(input_resolution, x):
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    H, W = input_resolution[0], input_resolution[1]
    x = x.view(B, H, W, C)
    x = x.permute(0, 3, 1, 2)
    return x


def ImgtoSeq(x):
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x = x.view(N, C, H * W)
    x = x.permute(0, 2, 1)
    return x


class feature_extractor(nn.Module):
    # torch image: C X H X W
    def __init__(self, in_chans=3, img_size=(384, 512), patch_size=(4, 4), embed_dim=48,
                 window_size=[6, 8], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True, **kwargs):
        """
        Learning the flow map which is utilized for Distorted Image Rectification
        :param in_chans: (int) Number of input image channels. Default: 3
        :param img_size: (int | tuple(int)) Input image size. Default: 384 x 512
        :param patch_size:  (int | tuple(int)) Patch size. Default: 4 x 4
        :param embed_dim: (int) Patch embedding dimension. Default: 48
        :param window_size: (int) Window size. Default: (12,16)
        :param mlp_ratio:  (int) Ratio of mlp hidden dim to embedding dim. Default: 4
        :param qkv_bias: (bool) If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: (float) Override default qk scale of head_dim ** -0.5 if set. Default: None
        :param drop_rate: (float) Dropout rate. Default: 0
        :param norm_layer:  (nn.Module) Normalization layer. Default: nn.LayerNorm.
        :param ape: (bool) If True, add absolute position embedding to the patch embedding. Default: False
        :param patch_norm: (bool) If True, add normalization after patch embedding. Default: True
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.pos_drop = nn.Dropout(p=drop_rate)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.partial_patch_embed = PartialPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                                     embed_dim=embed_dim,
                                                     norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            # Gaussian distribution: mean = 0, delta = 1, Now: mean =
            parameter = torch.randn(1, self.patch_embed.num_patches, embed_dim)
            parameter = (parameter - parameter.mean()) / parameter.std()
            parameter = parameter * 0.02
            self.absolute_pos_embed = nn.Parameter(parameter)

        # The core structure of our Transformer
        input_resolution0 = patches_resolution
        # Stage 0
        self.stage0 = PartialTransformerBlock(embed_dim, input_resolution0, num_heads=3,
                                              window_size=window_size,
                                              shift_size=0,
                                              mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                              drop_path=0.,
                                              act_layer=F.gelu, norm_layer=nn.LayerNorm)

        # Stage 1
        self.patch_merge1 = PatchMerging(input_resolution0, embed_dim, norm_layer=nn.LayerNorm)
        self.input_resolution1 = [input_resolution0[0] // 2, input_resolution0[1] // 2]
        self.stage1 = SwinTransformerBlock(embed_dim * 2, self.input_resolution1, num_heads=6,
                                           window_size=window_size,
                                           shift_size=0,
                                           mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                           drop_path=0.,
                                           act_layer=F.gelu, norm_layer=nn.LayerNorm)

        # Stage 2
        self.patch_merge2 = PatchMerging(self.input_resolution1, embed_dim * 2, norm_layer=nn.LayerNorm)
        self.input_resolution2 = [self.input_resolution1[0] // 2, self.input_resolution1[1] // 2]
        self.stage2 = SwinTransformerBlock(embed_dim * 4, self.input_resolution2, num_heads=12,
                                           window_size=window_size,
                                           shift_size=0,
                                           mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                           drop_path=0.,
                                           act_layer=F.gelu, norm_layer=nn.LayerNorm)

    def forward(self, image, mask):
        '''

        :param image: B*3*384*512
        :param mask: B*1*384*512
        :return:
        '''
        x = self.patch_embed(image)  # B*(96*128)*48
        patial_x = self.partial_patch_embed(image, mask)
        if self.ape:
            x = x + self.absolute_pos_embed
            patial_x = patial_x + self.absolute_pos_embed

        x = self.pos_drop(x)
        patial_x = self.pos_drop(patial_x)

        # Let the features pass the core structure of Transformer
        # Forward stage 0
        x0 = self.stage0(x, patial_x)  # B*(96*128)*48

        # Forward stage 1
        x1 = self.patch_merge1(x0)  # B*(48*64)*96
        x1 = self.stage1(x1)  # B*(48*64)*96

        # Forward stage 2
        x2 = self.patch_merge2(x1)  # B*(24*32)*192
        output = self.stage2(x2)  # B*(24*32)*192

        output = Seqtoimg(self.input_resolution2, output)  # B*192*24*32
        return output


class regression_Net(nn.Module):
    def __init__(self, grid_h, grid_w, embed_dim=48 * 4, window_size=[3, 4], mlp_ratio=4., qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0., patch_norm=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.input_resolution = [24, 32]

        self.patch_merge0 = PatchMerging(self.input_resolution, embed_dim, norm_layer=nn.LayerNorm)
        self.input_resolution0 = [self.input_resolution[0] // 2, self.input_resolution[1] // 2]
        self.stage0 = SwinTransformerBlock(embed_dim * 2, self.input_resolution0, num_heads=12,
                                           window_size=window_size,
                                           shift_size=0,
                                           mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                           drop_path=0.,
                                           act_layer=F.gelu, norm_layer=nn.LayerNorm)

        self.patch_merge1 = PatchMerging(self.input_resolution0, embed_dim * 2, norm_layer=nn.LayerNorm)
        self.input_resolution1 = [self.input_resolution0[0] // 2, self.input_resolution0[1] // 2]
        self.stage1 = SwinTransformerBlock(embed_dim * 4, self.input_resolution1, num_heads=12,
                                           window_size=window_size,
                                           shift_size=0,
                                           mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                           drop_path=0.,
                                           act_layer=F.gelu, norm_layer=nn.LayerNorm)

        self.patch_merge2 = PatchMerging(self.input_resolution1, embed_dim * 4, norm_layer=nn.LayerNorm)
        self.input_resolution2 = [self.input_resolution1[0] // 2, self.input_resolution1[1] // 2]
        self.stage2 = SwinTransformerBlock(embed_dim * 8, self.input_resolution2, num_heads=12,
                                           window_size=window_size,
                                           shift_size=0,
                                           mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                           drop_path=0.,
                                           act_layer=F.gelu, norm_layer=nn.LayerNorm)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dim * 8, out_channels=2048, kernel_size=[3, 4], padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     # nn.AdaptiveAvgPool2d(1),
        #     # 2048*1*1
        #     nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, padding=0, bias=True),
        #     # nn.Conv2d(in_channels=embed_dim * 8, out_channels=1024, kernel_size=1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     # 1024*1*1
        #     nn.Conv2d(in_channels=1024, out_channels=(grid_w + 1) * (grid_h + 1) * 2, kernel_size=1, padding=0,
        #               bias=True),
        #     # (U+1)*(V+1)*2
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim * 8, out_channels=1024, kernel_size=[3, 4], padding=0, bias=True),
            nn.ReLU(inplace=True),
            # 1024*1*1
            nn.Conv2d(in_channels=1024, out_channels=(grid_w + 1) * (grid_h + 1) * 2, kernel_size=1, padding=0,
                      bias=True),
            # (U+1)*(V+1)*2
        )

    def forward(self, x):
        # Forward stage 0
        x0 = ImgtoSeq(x)  # B*(24*32)*192
        x0 = self.patch_merge0(x0)  # B*(12*16)*384
        x0 = self.stage0(x0)  # B*(12*16)*384

        # Forward stage 1
        x1 = self.patch_merge1(x0)  # B*(6*8)*768
        x1 = self.stage1(x1)  # B*(6*8)*768

        # Forward stage 2
        x2 = self.patch_merge2(x1)  # B*(3*4)*1536
        x2 = self.stage2(x2)  # B*(3*4)*1536

        x3 = Seqtoimg(self.input_resolution2, x2)  # B*1536*3*4
        out = self.conv(x3)

        return out.view(-1, 2, self.grid_h + 1, self.grid_w + 1)


class RectanglingNetwork(nn.Module):
    # torch image: C X H X W
    def __init__(self, n_colors, grid_h, grid_w, width=512., height=384., cuda_flag=False, ite_num=1):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.width = width
        self.height = height
        # bulit model
        self.feature_extract = feature_extractor(n_colors)
        self.regressor_coarse = regression_Net(grid_h=grid_h, grid_w=grid_w)
        # self.regressor_fine = regression_Net(grid_h=grid_h, grid_w=grid_w)
        self.cuda_flag = cuda_flag
        self.ite_num = ite_num

    def forward(self, train_input, train_mask1, train_flag=True):
        train_mask = train_mask1[:, 0, :, :].unsqueeze(1)
        # features = self.feature_extract(torch.concat([train_input, train_mask], 1))
        features = self.feature_extract(train_input, train_mask)

        # feature = F.interpolate(features, size=(24, 32), mode='bilinear')
        feature = features

        # feature_map = torch.mean(features.squeeze(), 0).cpu().numpy()
        # plt.imshow(feature_map, cmap='viridis')  # 使用viridis颜色映射
        # plt.axis('off')
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # # plt.colorbar()  # 显示颜色条
        # plt.savefig('feature_map.png')  # 保存图片
        # plt.show()  # 显示图像

        # mesh_shift_total = []
        mesh_shift_primary = self.regressor_coarse(feature)
        # mesh_shift_total.append(mesh_shift_primary)
        if self.ite_num > 0:
            mesh_shift_final = mesh_shift_primary.clone()
            mesh_primary_local = shift2mesh(mesh_shift_primary / 16, 32., 24., grid_w=self.grid_w, grid_h=self.grid_h,
                                            cuda=self.cuda_flag)

            feature_warp = spatial_transform_local_feature.transformer(feature, mesh_primary_local, grid_w=self.grid_w,
                                                                       grid_h=self.grid_h, cuda=self.cuda_flag)
            # feature_map = torch.mean(feature_warp.squeeze(), 0).cpu().numpy()
            # plt.imshow(feature_map, cmap='viridis')  # 使用viridis颜色映射
            # plt.axis('off')
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # # plt.colorbar()  # 显示颜色条
            # plt.savefig('feature_map.png')  # 保存图片
            # plt.show()  # 显示图像

        if train_flag or self.ite_num == 0:
            mesh_primary = shift2mesh(mesh_shift_primary, width=self.width, height=self.height, grid_w=self.grid_w,
                                      grid_h=self.grid_h, cuda=self.cuda_flag)
            warp_image_primary, warp_mask_primary = spatial_transform_local.transformer(train_input, train_mask,
                                                                                        mesh_primary,
                                                                                        grid_w=self.grid_w,
                                                                                        grid_h=self.grid_h,
                                                                                        cuda=self.cuda_flag)
            if self.ite_num == 0:
                if train_flag:
                    return None, None, None, mesh_primary, warp_image_primary, warp_mask_primary
                else:
                    return warp_image_primary, warp_mask_primary

        for i in range(self.ite_num):
            # mesh_shift_error = self.regressor_fine(feature_warp)
            mesh_shift_error = self.regressor_coarse(feature_warp)
            # mesh_shift_total.append(mesh_shift_error)
            mesh_shift_final += mesh_shift_error
            if i < self.ite_num - 1:
                # mesh_shift_final = sum(mesh_shift_total)
                mesh_final_local = shift2mesh(mesh_shift_final / 16, 32., 24., grid_w=self.grid_w,
                                              grid_h=self.grid_h,
                                              cuda=self.cuda_flag)

                feature_warp = spatial_transform_local_feature.transformer(feature, mesh_final_local,
                                                                           grid_w=self.grid_w,
                                                                           grid_h=self.grid_h, cuda=self.cuda_flag)

        # feature_map = torch.mean(feature_warp.squeeze(), 0).cpu().numpy()
        # plt.imshow(feature_map, cmap='viridis')  # 使用viridis颜色映射
        # plt.axis('off')
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # # plt.colorbar()  # 显示颜色条
        # plt.savefig('feature_map.png')  # 保存图片
        # plt.show()  # 显示图像
        mesh_final = shift2mesh(mesh_shift_final, width=self.width, height=self.height,
                                grid_w=self.grid_w, grid_h=self.grid_h, cuda=self.cuda_flag)
        warp_image_final, warp_mask_final = spatial_transform_local.transformer(train_input, train_mask, mesh_final,
                                                                                grid_w=self.grid_w, grid_h=self.grid_h,
                                                                                cuda=self.cuda_flag)
        if train_flag:
            return mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final
        else:
            return warp_image_final, warp_mask_final


if __name__ == '__main__':
    train_input = 2 * torch.rand(3, 3, 384, 512) - 1
    train_mask = 2 * torch.rand(3, 3, 384, 512) - 1
    model = RectanglingNetwork(width=512., height=384., grid_w=8, grid_h=6)
    mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final = model(
        train_input, train_mask)
    print(mesh_primary.shape, warp_image_primary.shape, warp_mask_primary.shape, mesh_final.shape,
          warp_image_final.shape, warp_mask_final.shape)
