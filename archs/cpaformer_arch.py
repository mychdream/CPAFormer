import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
import math
from timm.models.layers import trunc_normal_


class MSCA(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1

        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

        self.reduction1 = nn.Conv2d(dim1, dim1, kernel_size=2, stride=2, groups=dim1)
        self.dwconv = nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, groups=dim1)
        self.conv = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(dim1),
            nn.GELU())

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

    def forward(self, x, h, w):

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)  # B,num_heads,N,1
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        _x = x_
        for _ in range(4):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, C, -1).permute(0, 2, 1)  # B,N/16/16,C
        _x = self.norm_act(_x)
        x_ = _x
        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)  # B,num_heads,N,1
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                   3)  # B,num_heads,N,C//num_heads

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,num_heads,N,N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B,num_heads,N,C//num_heads

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DSFI(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.conv = nn.Sequential(nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_features // 5, out_features, 1, 1, 0))

        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, h, w):
        B, L, C = x.shape
        H, W = h, w
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.conv(x) * self.linear(x)

        return x


class DAIT(nn.Module):
    def __init__(self, dim, mlp_dim, heads):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))

        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, 2 * dim, 1, 1, 0)
        self.cra = MSCA(dim1=dim, num_heads=heads)
        self.DSFI = DSFI(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        x = self.DSFI(x, h, w)
        x = self.conv2(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = self.dwc(x2)
        x2 = rearrange(x2, 'b c h w->b (h w) c')
        y = self.cra(x1, h, w)
        y = y + x2
        x = residual + y
        x = self.mlp(x, x_size=(h, w)) + x

        return rearrange(x, 'b (h w) c->b c h w', h=h)

# Patch Dividing Function
def divide_patches(x, patch_size, stride):
    b, c, h, w = x.size()
    patches = []
    n_rows = 0
    for i in range(0, h + stride - patch_size, stride):
        top = i
        bottom = i + patch_size
        if bottom > h:
            top = h - patch_size
            bottom = h
        n_rows += 1
        for j in range(0, w + stride - patch_size, stride):
            left = j
            right = j + patch_size
            if right > w:
                left = w - patch_size
                right = w
            patches.append(x[:, :, top:bottom, left:right])
    n_cols = len(patches) // n_rows
    patches = torch.stack(patches, dim=0)  # (num_patches, b, c, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3, 4).contiguous()  # (b, num_patches, c, patch_size, patch_size)
    return patches, n_rows, n_cols


# Patch Reverse Function
def reverse_patches(patches, x, stride, patch_size):
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + stride - patch_size, stride):
        top = i
        bottom = i + patch_size
        if bottom > h:
            top = h - patch_size
            bottom = h
        for j in range(0, w + stride - patch_size, stride):
            left = j
            right = j + patch_size
            if right > w:
                left = w - patch_size
                right = w
            output[:, :, top:bottom, left:right] += patches[:, index]
            index += 1

    # Normalize overlapping regions
    for i in range(stride, h + stride - patch_size, stride):
        top = i
        bottom = i + patch_size - stride
        if top + patch_size > h:
            top = h - patch_size
        output[:, :, top:bottom, :] /= 2
    for j in range(stride, w + stride - patch_size, stride):
        left = j
        right = j + patch_size - stride
        if left + patch_size > w:
            left = w - patch_size
        output[:, :, :, left:right] /= 2

    return output


# Convolutional Feed-Forward Network (ConvFFN)
class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, kernel_size=5, activation=nn.GELU):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.activation = activation()
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, groups=hidden_channels),
            nn.GELU()
        )
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, patch_size):
        x = self.fc1(x)
        x = self.activation(x)
        x = x + self.dwconv(x, patch_size)
        x = self.fc2(x)
        return x


class Bk(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Scope: Attention Layer
class Scope(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


# HST Module
class HST(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()
        self.attn_layer = Bk(dim, Scope(dim, heads, qk_dim))
        self.ffn_layer = Bk(dim, ConvFFN(dim, mlp_dim))

    def forward(self, x, patch_size):
        stride = patch_size - 2
        patches, n_rows, n_cols = divide_patches(x, patch_size, stride)
        b, n, c, ph, pw = patches.shape

        # Attention layer
        patches = rearrange(patches, 'b n c h w -> (b n) (h w) c')
        patches = self.attn_layer(patches) + patches
        patches = rearrange(patches, '(b n) (h w) c -> b n c h w', n=n_cols, w=pw)

        # Reverse patches to original image
        x = reverse_patches(patches, x, stride, patch_size)

        # Feedforward layer
        b, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ffn_layer(x, patch_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)

        return x


@ARCH_REGISTRY.register()
class CPAFormer(nn.Module):
    def __init__(self, in_chans=3, upscale=2, dim=64, block_num=8, qk_dim=36, mlp_dim=96, heads=4,
                 patch_sizes=[16, 20, 24, 28, 16, 20, 24, 28]):
        super(CPAFormer, self).__init__()

        self.dim = dim
        self.block_num = block_num
        self.patch_sizes = patch_sizes
        self.qk_dim = qk_dim
        self.mlp_dim = mlp_dim
        self.upscale = upscale
        self.heads = heads

        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()

        for i in range(self.block_num):
            self.blocks.append(nn.ModuleList([DAIT(self.dim, self.qk_dim, heads=self.heads),
                                              HST(self.dim, self.qk_dim,
                                                  self.mlp_dim, self.heads)]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Pass the input through the feature extraction layers."""
        for i in range(self.block_num):
            residual = x
            global1, local1 = self.blocks[i]
            x = global1(x)  # Global attention
            x = local1(x, self.patch_sizes[i])  # Local attention (patch-based)
            x = residual + self.mid_convs[i](x)
        return x

    def forward(self, x):
        """Forward pass of the network."""
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False) if self.upscale != 1 else x
        x = self.first_conv(x)
        x = self.forward_features(x) + x

        # Upscaling
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))

        out = self.last_conv(out) + base
        return out

    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
                                                   num_parameters / 10 ** 3)
