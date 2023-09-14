"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F

import torch
import numpy as np



class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False

    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size // 2,
                        groups=self.dim)  # equal to x + conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True,
                 downsample=False, kernel_size=5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        x = self.fc1(x) # 1x1conv
        x = self.act1(x) #nn.GELU
        # x = self.drop(x)
        x = self.conv(x)# 3x3conv
        x = self.fc2(x)# 1x1conv
        # x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2) #weight – filters of shape(out_channels, in_channel/gounps,kh,kw)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        DD, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (DD, ww))  # (B, C, DD, ww) 初始化Super Token
        pixel_features = x.reshape(B, C, DD, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, DD * ww, h * w, C)#reshape tokens

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, DD*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, DD * ww, C, 9) #转置矩阵
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, DD*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1)  # (B, DD*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, DD, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, DD*ww, C, 9)
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, DD, ww)).reshape(
                        B, C, DD, ww)
                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, DD, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, DD*ww, C, 9) torch.Size([1, 196, 64, 9])
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, DD, ww)).reshape(B, C, DD, ww)
        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, DD, ww)

        stoken_features = self.stoken_refine(stoken_features) #MHSA
        stoken_features = self.unfold(stoken_features)  # (B, C*9, DD*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, DD * ww, C, 9)  # (B, DD*ww, C, 9)
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, DD*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, DD, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, n_iter, stoken_size,
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values=1.0e-5):
        super().__init__()

        self.layerscale = layerscale

        self.pos_embed = ResDWC(dim, 3)
        self.pool1 = WavePool(64)
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size=stoken_size,
                                    n_iter=n_iter,
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
                        drop=drop)

        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        x = self.pos_embed(x)  # CPE Position Embedding (CPE)

        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))  # Super Token Attention (STA)

            # x = x + self.drop_path(self.mlp2(self.norm2(x)))  # Convolutional Feed-Forward-Network (ConvFFN)
        return x

class BasicLayer(nn.Module):
    def __init__(self, num_layers, dim, n_iter, stoken_size,
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values=1.0e-5,
                 downsample=False,
                 use_checkpoint=False, checkpoint_num=None):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num

        self.blocks = nn.ModuleList([StokenAttentionLayer(
            dim=dim[0], n_iter=n_iter, stoken_size=stoken_size,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            act_layer=act_layer,
            layerscale=layerscale, init_values=init_values) for i in range(num_layers)])

        if downsample:
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None

    def forward(self, x):
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # if self.downsample is not None:
        #     x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),

            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),

        )

    def forward(self, x):
        x = self.proj(x)
        return x



class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x



class MODEL(nn.Module):
    def __init__(self, in_chans=1,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 n_iter=[3, 2, 1, 0], stoken_size=[8, 4, 2, 1],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 projection=None, freeze_bn=False,
                 use_checkpoint=False, checkpoint_num=[0, 0, 0, 0],
                 layerscale=[False, False, False, False], init_values=1e-6, option_unpool='sum', stoken_refine=True,
        stoken_refine_attention=True,
        hard_label=False,
        rpe=False,):
        super().__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0_0 = nn.Conv2d(1, 3, 1, 1, 0)
        # self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 320, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(320, 320, 3, 1, 0)
        self.pool3 = WavePool(320)

        self.conv4_1_e = nn.Conv2d(320, 512, 3, 1, 0)



        # self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        self.freeze_bn = freeze_bn

        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])
        self.pos_drop = nn.Dropout(p=drop_rate)
        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        self.EN1 = BasicLayer(num_layers=depths[0],
                                dim=[embed_dim[0],
                                     embed_dim[1] if 0 < self.num_layers - 1 else None],
                                n_iter=n_iter[0],
                                stoken_size=to_2tuple(stoken_size[0]),
                                num_heads=num_heads[0],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                downsample=0 < self.num_layers - 1,
                                use_checkpoint=use_checkpoint,
                                checkpoint_num=checkpoint_num[0],
                                layerscale=layerscale[0],
                                init_values=init_values)
        self.downsample1 = PatchMerging(embed_dim[0], embed_dim[1])

        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #
        # self.norm1 = nn.BatchNorm2d(embed_dim[0])
        # self.mlp1 = Mlp(in_features=embed_dim[0], hidden_features=int(embed_dim[0] * mlp_ratio), out_features=embed_dim[0], act_layer=nn.GELU,
        #                 drop=drop_rate)

        self.EN2 = BasicLayer(num_layers=depths[1],
                           dim=[embed_dim[1],
                                embed_dim[1 + 1] if 1 < self.num_layers - 1 else None],
                           n_iter=n_iter[1],
                           stoken_size=to_2tuple(stoken_size[1]),
                           num_heads=num_heads[1],
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                           downsample=1 < self.num_layers - 1,
                           use_checkpoint=use_checkpoint,
                           checkpoint_num=checkpoint_num[1],
                           layerscale=layerscale[1],
                           init_values=init_values)
        self.downsample2 = PatchMerging(embed_dim[1], embed_dim[2])
        self.EN3 = BasicLayer(num_layers=depths[2],
                           dim=[embed_dim[2],
                                embed_dim[2 + 1] if 2 < self.num_layers - 1 else None],
                           n_iter=n_iter[2],
                           stoken_size=to_2tuple(stoken_size[2]),
                           num_heads=num_heads[2],
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                           downsample=2 < self.num_layers - 1,
                           use_checkpoint=use_checkpoint,
                           checkpoint_num=checkpoint_num[2],
                           layerscale=layerscale[2],
                           init_values=init_values)
        self.downsample3 = PatchMerging(embed_dim[2], embed_dim[3])
        self.EN4 = BasicLayer(num_layers=depths[3],
                           dim=[embed_dim[3],
                                embed_dim[3 + 1] if 3 < self.num_layers - 1 else None],
                           n_iter=n_iter[3],
                           stoken_size=to_2tuple(stoken_size[3]),
                           num_heads=num_heads[3],
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                           downsample=3 < self.num_layers - 1,
                           use_checkpoint=use_checkpoint,
                           checkpoint_num=checkpoint_num[3],
                           layerscale=layerscale[3],
                           init_values=init_values)

        for i_layer in range(self.num_layers):
            layer = BasicLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer],
                                    embed_dim[i_layer + 1] if i_layer < self.num_layers - 1 else None],
                               n_iter=n_iter[i_layer],
                               stoken_size=to_2tuple(stoken_size[i_layer]),
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=i_layer < self.num_layers - 1,
                               use_checkpoint=use_checkpoint,
                               checkpoint_num=checkpoint_num[i_layer],
                               layerscale=layerscale[i_layer],
                               init_values=init_values)
            self.layers.append(layer)

        self.proj = nn.Conv2d(self.num_features, projection, 1) if projection else None
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(projection or self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # 4个3 × 3卷积
        x = self.pos_drop(x)
        x1 = self.EN1(x)
        layer_list = []
        for layer in self.layers:
            x = layer(x)
            layer_list.append(x)

        return layer_list

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x, skips

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if self.option_unpool == 'sum':
            layer_list=[]
            if level == 1:
                # out = self.patch_embed(x)  # 4个3 × 3卷积
                # out = self.pos_drop(out)  # (64,224,224)
                out = self.conv0_0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                out = self.relu(self.conv1_2(self.pad(out)))

                LL, VD, HD, DD = self.pool1(out)
                skips['pool1'] = [VD, HD, DD]


                return LL #(64,112,112)

            elif level == 2:
                out =self.EN1(x)#x->(64,112,112), out->(64,112,112)
                LL, VD, HD, DD = self.pool1(out) # out->(128,112,112)
                skips['pool2'] = [VD, HD, DD]
                LL = self.downsample1(LL)
                return LL #(128,56,56)

            elif level == 3:

                out = self.EN2(x)#out->(128,56,56)
                LL, VD, HD, DD = self.pool2(out)#x->(320,56,56), out->(320,56,56)
                skips['pool3'] = [VD, HD, DD]
                LL = self.downsample2(LL)
                return LL#(320,28,28)
            else:
                return self.relu(self.conv4_1_e(self.pad(x)))

        else:
            raise NotImplementedError


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_VD = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HD = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_DD = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_VD = torch.from_numpy(harr_wav_VD).unsqueeze(0)
    filter_HD = torch.from_numpy(harr_wav_HD).unsqueeze(0)
    filter_DD = torch.from_numpy(harr_wav_DD).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    VD = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HD = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    DD = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    VD.weight.requires_grad = False
    HD.weight.requires_grad = False
    DD.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    VD.weight.data = filter_VD.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HD.weight.data = filter_HD.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    DD.weight.data = filter_DD.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, VD, HD, DD


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.VD, self.HD, self.DD = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.VD(x), self.HD(x), self.DD(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.VD, self.HD, self.DD = get_wav(self.in_channels, pool=False)

    def forward(self, LL, VD, HD, DD, original=None):
        if self.option_unpool == 'sum':
            shape_x1 = LL.size()
            shape_x2 = VD.size()
            left = 0
            right = 0
            top = 0
            bot = 0
            if shape_x1[3] != shape_x2[3]:
                lef_right = shape_x2[3] - shape_x1[3]
                if lef_right % 2 is 0.0:
                    left = int(lef_right / 2)
                    right = int(lef_right / 2)
                else:
                    left = int(lef_right / 2)
                    right = int(lef_right - left)

            if shape_x1[2] != shape_x2[2]:
                top_bot = abs(shape_x1[2] - shape_x2[2])
                if top_bot % 2 is 0.0:
                    top = int(top_bot / 2)
                    bot = int(top_bot / 2)
                else:
                    top = int(top_bot / 2)
                    bot = int(top_bot - top)
            reflection_padding = [left, right, top, bot]
            reflection_pad = nn.ReflectionPad2d(reflection_padding)
            LL = reflection_pad(LL)
            return self.LL(LL) + self.VD(VD) + self.HD(HD) + self.DD(DD)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.VD(VD), self.HD(HD), self.DD(DD), original], dim=1)
        else:
            raise NotImplementedError

class WaveDecoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool

        if option_unpool == 'sum':
            multiply_in = 1
        elif option_unpool == 'cat5':
            multiply_in = 5
        else:
            raise NotImplementedError

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 320, 3, 1, 0)
        self.sig=nn.Sigmoid()
        self.recon_block3 = WaveUnpool(320, option_unpool)
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(320*multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(320*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = WaveUnpool(128, option_unpool)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv0_1 = nn.Conv2d(64, 1, 3, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))

            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            VD, HD, DD = skips['pool3']
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = self.recon_block2(out, VD, HD, DD, original)
            _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))

        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            VD, HD, DD = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = self.recon_block1(out, VD, HD, DD, original)
            _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            out=self.conv1_1(self.pad(x))
            VD, HD, DD = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = self.recon_block1(out, VD, HD, DD, original)

            return [self.sig(self.conv0_1(self.pad(out)))]
