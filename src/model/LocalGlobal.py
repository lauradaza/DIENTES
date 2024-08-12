import numpy as np
from typing import Sequence, Type
from collections.abc import Sequence

import torch
from torch import nn, einsum
from torch.nn import LayerNorm
import torch.nn.functional as F

from monai.networks.blocks import PatchEmbed
from monai.networks.blocks import MLPBlock as Mlp

from monai.utils import optional_import
from monai.networks.layers import DropPath, trunc_normal_

from einops import rearrange, repeat

from src.model.utils import default, checkpoint, patchify_features, window_reverse
from src.model.posEmbed import PositionalEncodingPermute3D


rearrange, _ = optional_import("einops", name="rearrange")


class PatchMerging(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:  # type: ignore
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, d % 2, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 0::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowCrossAttention(nn.Module):
    """
    based on MONAI's implementation of Swin Transformer
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        window_size: Sequence[int],
        context_dim: int = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        xAttn: bool = False,
        temp: float = 0.1,
    ) -> None:
        """
        Args:
            query_dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            context_dim: number of feature channels (context).
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.query_dim = query_dim
        context_dim = default(context_dim, query_dim)
        self.context_dim = context_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = query_dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if not xAttn:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1)
                    * (2 * self.window_size[1] - 1)
                    * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            trunc_normal_(self.relative_position_bias_table, std=0.02)
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(
                    torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
                )
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("temp", torch.tensor([temp]))

    def forward(self, x, context=None, mask=None, cls_token=True):
        b, n, c = x.shape
        xAttn = context is not None
        if cls_token:
            n -= 1  # the first token is the patch_token

        Q = self.to_q(x) * self.scale  # b, n, c*h
        context = default(context, x)
        K = self.to_k(context)
        V = self.to_v(context)
        if len(context) != len(x):
            # expand the batch dimension
            expansion = len(x) // len(context)
            K = torch.repeat_interleave(K, expansion, dim=0)
            V = torch.repeat_interleave(V, expansion, dim=0)

        Q, K, V = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (Q, K, V),
        )

        attn = Q @ K.transpose(-2, -1)

        if not xAttn:  # self-attn
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.clone()[:n, :n].reshape(-1)
            ].reshape(
                n, n, -1
            )  # n, n, h
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn[:, :, -n:, -n:] = attn[
                :, :, -n:, -n:
            ] + relative_position_bias.unsqueeze(0)

        attn, V = map(lambda t: rearrange(t, "b h n d -> (b h) n d"), (attn, V))

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(attn.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=self.num_heads)
            attn.masked_fill_(~mask, max_neg_value)

        attn = F.softmax(attn / self.temp, dim=-1)

        attn = self.attn_drop(attn)
        x = einsum("b i j, b j d -> b i d", attn, V)
        x = rearrange(x, "(b h) n d -> b n (h d)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class normLayerOrder(nn.Module):
    def __init__(self, norm, layer, pre=True) -> None:
        super().__init__()
        self.norm = norm
        self.layer = layer

        if pre:
            self.forward = self._pre
        else:
            self.forward = self._post

    def _pre(self, x, **kwargs):
        x = self.layer(self.norm(x), **kwargs)
        return x

    def _post(self, x, **kwargs):
        x = self.norm(self.layer(x, **kwargs))
        return x


class LocalGlobalLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        window_size=(6, 6, 6),
        num_windows=(1, 8, 8, 8),
        mlp_ratio=4,
        ctx_attn=False,
        temp=1.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_windows = num_windows
        norm1 = nn.LayerNorm(dim)
        attn = WindowCrossAttention(
            query_dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            temp=temp,
        )  # is a self-attention
        self.Attn = normLayerOrder(norm1, attn, pre_norm)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = self.mlp = Mlp(
            hidden_size=dim,
            mlp_dim=int(dim * mlp_ratio),
            act="GELU",
            dropout_rate=attn_drop,
            dropout_mode="swin",
        )
        self.MLP = normLayerOrder(self.norm2, self.mlp, pre_norm)

        if ctx_attn:
            normCtx1 = nn.LayerNorm(dim)
            attnCtx1 = WindowCrossAttention(
                query_dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                temp=temp,
            )  # is self-attn if context is none
            self.CtxAttn1 = normLayerOrder(normCtx1, attnCtx1, pre_norm)
            self._forward_ctx = self._ctx_attn
        else:
            self._forward_ctx = nn.Identity()

    def _ctx_attn(self, x):
        context, x = x[:, 0], x[:, 1:]

        h, w, d = self.num_windows
        ch = context.shape[-1]

        context = context.view(-1, h * w * d, ch)
        context = self.CtxAttn1(context, cls_token=False) + context

        context = context.view(-1, 1, ch)

        x = torch.cat([context, x], dim=1)
        return x

    def forward(self, x):
        x = self.Attn(x) + x
        x = self._forward_ctx(x)
        x = self.MLP(x) + x
        return self.drop_path(x)  # TODO: context?


class LocalGlobalBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop_path=[0.0, 0.0],
        attn_drop=0.0,
        use_checkpoint=True,
        downsample=True,
        pre_norm=True,
        window_size=(6, 6, 6),
        num_windows=(8, 8, 8),
        mlp_ratio=4,
        temp=1.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_windows = torch.tensor(num_windows)
        self.checkpoint = use_checkpoint
        last = torch.all(self.num_windows == 1)
        skip_ctx = last or torch.prod(self.num_windows) > 256
        self.blocks = nn.ModuleList(
            [
                LocalGlobalLayer(
                    dim=dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    pre_norm=pre_norm,
                    window_size=self.window_size,
                    num_windows=num_windows,
                    mlp_ratio=mlp_ratio,
                    ctx_attn=(i + 1) % 2 == 1 and not skip_ctx,
                    temp=temp,
                )
                for i in range(len(drop_path))
            ]
        )

        if downsample:
            self.downsample = downsample(
                dim=dim, norm_layer=nn.LayerNorm, spatial_dims=3
            )
            self.downsample_ctx = downsample(
                dim=dim, norm_layer=nn.LayerNorm, spatial_dims=3
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x, context):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        # context: [B*N, 1, Ch]
        B, C, H, W, D = x.shape

        x, dims, window_size, [pad_d1, pad_b, pad_r] = patchify_features(
            x, self.window_size
        )  # [B*N, nWindows, Ch]

        x = torch.cat([context, x], dim=1)

        for blk in self.blocks:
            x = blk(x)

        context, x = x[:, 0].unsqueeze(0), x[:, 1:]

        x = window_reverse(x, window_size, dims)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :D, :].contiguous()
        context = window_reverse(
            context.unsqueeze(0), (1, 1, 1), (B, *self.num_windows)
        )

        x = self.downsample(x)  # TODO: check normalization order (?)
        x = rearrange(x, "b d h w c -> b c d h w")
        context = self.downsample_ctx(context)
        context = rearrange(context, "b d h w c -> (b d h w) 1 c")
        return x, context


class LocalGlobalTransformer(nn.Module):
    """
    Adapted from MONAI's SwinUNETR implementation
    """

    def __init__(
        self,
        in_chans: int,
        window_size: Sequence[int],
        embed_dim: int,
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        img_size: Sequence[int] = (96, 96, 96),
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_emb = PositionalEncodingPermute3D(embed_dim)

        num_windows = [
            int(np.ceil((isz // 2) / wsz)) for isz, wsz in zip(img_size, window_size)
        ]
        scale = embed_dim ** (-1 / 3)
        self.patch_token = nn.Parameter(
            scale * torch.randn(np.prod(num_windows), 1, embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        temps = torch.ones(sum(depths)) * 0.4
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_depth in range(len(depths)):
            i_dim = int(embed_dim * 2**i_depth)
            layer = LocalGlobalBlock(
                dim=i_dim,
                num_heads=num_heads[i_depth],
                drop_path=dpr[sum(depths[:i_depth]) : sum(depths[: i_depth + 1])],
                attn_drop=attn_drop_rate,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint,
                pre_norm=True,
                window_size=window_size,
                num_windows=[i // 2**i_depth for i in num_windows],
                mlp_ratio=mlp_ratio,
                temp=temps[i_depth].item(),
            )
            if i_depth == 0:
                self.layers1.append(layer)
            elif i_depth == 1:
                self.layers2.append(layer)
            elif i_depth == 2:
                self.layers3.append(layer)
            elif i_depth == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True, metadata=None):
        x0 = self.patch_embed(x)
        pos = self.pos_emb(x0, metadata["spacing"], metadata["corner"])

        x0 = self.pos_drop(x0 + pos)

        context = self.patch_token
        context = torch.repeat_interleave(context, x.shape[0], dim=0)

        x0_out = self.proj_out(x0, normalize)
        x1, context = self.layers1[0](x0.contiguous(), context)
        x1_out = self.proj_out(x1, normalize)
        x2, context = self.layers2[0](x1.contiguous(), context)
        x2_out = self.proj_out(x2, normalize)
        x3, context = self.layers3[0](x2.contiguous(), context)
        x3_out = self.proj_out(x3, normalize)
        x4, context = self.layers4[0](x3.contiguous(), context)
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out], context
