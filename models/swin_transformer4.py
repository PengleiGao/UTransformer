import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.selfattention1 import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSTM_small2(nn.Module):
    def __init__(self, pre_step, size=[768, 4, 4], device=0):
        super(LSTM_small2, self).__init__()

        self.channel, self.height, self.width = size[0], size[1] + int(2 * pre_step), int(size[2] + 2 * pre_step)
        self.L = self.height * self.width
        self.n_block = int(self.height / 2)
        self.split = self.width
        #self.emb = nn.Linear(16, self.L)
        self.lstm_size = self.channel * int(self.height / self.n_block)
        self.LSTM_encoder1 = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.LSTM_decoder1 = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        #self.LSTM_encoder2 = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        #self.LSTM_decoder2 = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.multiatt = MultiHeadAttention(self.channel, n_head=4, d_model=512)
        self.device = device
        self.norm_layer = nn.LayerNorm(self.channel)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.height, self.width, self.channel)
        init_hidden = (Variable(torch.zeros(2, x.shape[0], self.lstm_size)).cuda(self.device),
                       Variable(torch.zeros(2, x.shape[0], self.lstm_size)).cuda(self.device))

        xout_total = []
        for i in range(self.n_block):
            xblock = x[:, 2 * i:2 * (i + 1), :, :]
            xblock = torch.stack(torch.split(xblock, 1, dim=2)).view(self.split, B, 1, self.lstm_size)
            # Encode feature for small task
            en_hiddeni = init_hidden
            xsmall_out = []
            for j in reversed(range(self.split)):
                en_outi, en_hiddeni = self.LSTM_encoder1(xblock[j], en_hiddeni)

            # Decode feature for small task
            de_outi, de_hiddeni = en_outi, en_hiddeni
            for k in range(self.split):
                de_outi, de_hiddeni = self.LSTM_decoder1(de_outi, de_hiddeni)
                xsmall_out.append(de_outi.view(B, 2, 1, self.channel))
            xout_total.append(torch.cat(xsmall_out, dim=2))

        attout = []
        for i in range(self.n_block):
            if i==0:
                q, k, v = xout_total[i+1], xout_total[i+2], xout_total[i]
            elif i==self.n_block-1:
                q, k, v = xout_total[i-1], xout_total[i-2], xout_total[i]
            else:
                q, k, v = xout_total[i-1], xout_total[i+1], xout_total[i]
            q, k, v = q.view(B, -1, self.channel), k.view(B, -1, self.channel), v.view(B, -1, self.channel)
            attouti = self.multiatt(q, k, v).view(B, 2, self.width, self.channel)
            attout.append(attouti)
        feature1 = torch.cat(attout, dim=1)

        # x_out2 = []
        # en_hidden2 = init_hidden
        # for i in range(self.split):
        #     en_out2, en_hidden2 = self.LSTM_encoder2(x_split[i], en_hidden2)
        #
        # de_out2, de_hidden2 = en_out2, en_hidden2
        # for j in reversed(range(self.split)):
        #     de_out2, de_hidden2 = self.LSTM_decoder2(de_out2, de_hidden2)
        #     x_out2.append(de_out2.view(B, self.height, 1, self.channel))
        # feature2 = torch.cat(x_out2, dim=2)

        #feat = (feature1 + feature2) / 2
        feature = feature1.view(B, -1, self.channel)
        feature = self.norm_layer(feature)
        out_decode = feature.view(B, self.height, self.width, self.channel)
        out_decode = out_decode[:, 1:5, 1:5, :]

        return feature, out_decode


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
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
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

        # if self.shift_size > 0:
        #     # calculate attention mask for SW-MSA
        #     H, W = self.input_resolution
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1
        #
        #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # else:
        #     attn_mask = None
        #
        # self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        #H, W = self.input_resolution
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchMerging_reverse(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.unreduction = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(int(dim/2))

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        #H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = self.unreduction(x).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)

        x = x.view(B, int(C/2), -1)  # B C/4 (H*2)*(W*2)
        x = x.permute(0, 2, 1)

        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, mode=None):

        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.mode = mode

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
                Args:
                    x: Input feature, tensor size (B, H*W, C).
                    H, W: Spatial resolution of the input feature.
                """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if self.mode=='encode':
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            elif self.mode=='decode':
                Wh, Ww = int(H * 2), int(W * 2)
            else:
                print("wrong mode")
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class SwinTransformer4(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pre_step, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.pretrain_img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.pre_step = pre_step
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # self.num_features = num_features

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.MRB = LSTM_small2(pre_step)
        self.patch_embed_reversed = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim, 48, kernel_size=5, stride=1, padding=2),
            LayerNorm(48),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(48, 24, kernel_size=5, stride=1, padding=2),
            LayerNorm(24),
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 3, kernel_size=1, stride=1, padding=0)
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint, mode='encode')
            self.layers.append(layer)

        self.decoder_layers = nn.ModuleList()
        for j_layer in reversed(range(self.num_layers)):
            de_layer = BasicLayer(dim=int(embed_dim * 2 ** j_layer),
                               depth=depths[j_layer],
                               num_heads=num_heads[j_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:j_layer]):sum(depths[:j_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging_reverse if (j_layer > 0) else None,
                               use_checkpoint=use_checkpoint, mode='decode')
            self.decoder_layers.append(de_layer)

        self.norm = norm_layer(self.num_features)
        self.norm_de = norm_layer(self.embed_dim)
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def encode_features(self, x):
        #x = F.pad(x,(32,32,32,32),'constant',1)
        x = self.patch_embed(x)
        shortcut = []

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for layer in self.layers:
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            shortcut.append(x_out)

        out = self.norm(x_out)  # B L C
        #x = self.avgpool(x.transpose(1, 2))  # B C 1
        #x = torch.flatten(x, 1)
        out_encode = out.view(out.size(0), H, W, self.num_features)
        return out, shortcut, out_encode

    def decode_features(self, x, shortcut_x):
        B = x.size(0)
        Wh = int(math.sqrt(x.size(1)))
        Ww = Wh
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C

        #x = self.pos_drop(x)
        i = 0

        for layer in self.decoder_layers:
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i < 3:
                x = x.view(B, Wh, Ww, -1)
                #shortcutx = shortcut_x[2 - i].view(B, 16 * 2 ** i, 16 * 2 ** i, -1)[:, 2 ** (i + 1):14 * 2 ** i,
                #            2 ** (i + 1):14 * 2 ** i, :]
                #shortcutx = shortcut_x[2 - i].view(B, 16 * 2 ** i, 16 * 2 ** i, -1)[:, 2 ** (i + 2):3 * 2 ** (i + 2),
                #            2 ** (i + 2):3 * 2 ** (i + 2), :]
                shortcutx = shortcut_x[2 - i].view(B, Wh, Wh, -1)[:, 2**(i+1):10*2**i,2**(i+1):10*2**i,:]
                #shortcutx = shortcut_x[2 - i].view(B, Wh-2**(i+2), Wh-2**(i+2), -1)[:, 2 ** (i + 1):10 * 2 ** i, 2 ** (i + 1):10 * 2 ** i,:]

                merge = (x[:, 2**(i+1):10*2**i,2**(i+1):10*2**i,:] + shortcutx)/2
                x[:, 2**(i+1):10*2**i,2**(i+1):10*2**i,:] = merge
                #merge=(x[:,2**(i+2):3*2**(i+2),2**(i+2):3*2**(i+2),:] + shortcutx) / 2
                #x[:,2**(i+2):3*2**(i+2),2**(i+2):3*2**(i+2),:] = merge
                #merge = (x[:,2**(i+1):14*2**i,2**(i+1):14*2**i,:] + shortcutx) / 2
                #x[:,2**(i+1):14*2**i,2**(i+1):14*2**i,:] = merge
                x = x.view(B, Wh*Ww, -1)
                i = i + 1


        out = self.norm_de(x_out)
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.patch_embed_reversed(out)
        return out

    def forward(self, x, only_encode=False):
        B = x.size(0)
        x, shortcut_x, out_encode = self.encode_features(x)
        if only_encode:
            return out_encode
        x, out_decode = self.MRB(x)
        #x = x.view(B, 8, 8, self.num_features)
        x = x.view(B, 4+2*self.pre_step, 4+2*self.pre_step, self.num_features)

        x[:, 1:5, 1:5, :] = shortcut_x[3].view(B, 4+2*self.pre_step, 4+2*self.pre_step, self.num_features)[:, 1:5, 1:5, :]
        #iner=shortcut_x[3].view(B,4+2*self.pre_step,4+2*self.pre_step,self.num_features)[:,1:7,1:7,:]
        #x[:,1:7,1:7,:]=iner
        x = x.view(B, -1, self.num_features)
        out = self.decode_features(x, shortcut_x)
        return out, out_decode

