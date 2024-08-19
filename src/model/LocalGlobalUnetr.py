import numpy as np
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.utils import ensure_tuple_rep, optional_import
from monai.networks.blocks import (
    UnetrBasicBlock,
    UnetrUpBlock,
)
from monai.networks.blocks.dynunet_block import get_conv_layer

from src.model.LocalGlobal import LocalGlobalTransformer


rearrange, _ = optional_import("einops", name="rearrange")


class CrossAttnUnetr(nn.Module):
    """
    Based on MONAI's Swin UNETR implementation
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        entities: bool = False,
        clip_dim: int = 256,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            entities: wether to use standard semantic segmentation or dynamic heads
            clip_dim: if word_embedding, dim of the encoding
        """

        super().__init__()

        self.out_channels = out_channels

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(6, spatial_dims)
        self.window_size = window_size

        # sanity checks
        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p, w in zip(img_size, patch_size, window_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError(
                        "input image size (img_size) should be divisible by stage-wise image resolution."
                    )
                if (m // np.power(p, i + 1)) % w != 0 and i < 4:
                    print((m // np.power(p, i + 1)), w)
                    raise ValueError(
                        "window_size should be divisible by stage-wise image resolution."
                    )

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        # Encoder
        self.swinViT = LocalGlobalTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            img_size=img_size,
        )

        # Bottleneck
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Skip connections
        mult = [1, 1, 2, 4]
        for i in range(1, 5):
            in_channels_temp = in_channels if i == 1 else feature_size
            setattr(
                self,
                "encoder" + str(i),
                UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=mult[i - 1] * in_channels_temp,
                    out_channels=mult[i - 1] * feature_size,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True,
                ),
            )

        # Decoder
        mult_in = 16
        for i in range(5):
            mult_out = mult_in // 2 if i < 4 else mult_in
            block_number = str(5 - i)  # decoder block numbers go backwards
            setattr(
                self,
                "decoder" + block_number,
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=mult_in * feature_size,
                    out_channels=mult_out * feature_size,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True,
                ),
            )
            mult_in = mult_out

        # Output head for semantic segmentation
        self.output = get_conv_layer(
            spatial_dims,
            feature_size if not entities else 8,
            out_channels if not entities else 1,
            kernel_size=1,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )
        self.forward_output = self.forward_output_semseg

        if entities:
            # adjust decoder's output size
            self.precls_conv = nn.Sequential(
                nn.Conv3d(feature_size, feature_size, kernel_size=1),
                nn.GroupNorm(16, feature_size),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )

            # dinamic conv
            weight_nums, bias_nums = [], []
            weight_nums.append(feature_size * 8)  # in_ch * out_ch
            bias_nums.append(8)  # out_ch

            assert len(weight_nums) == len(bias_nums)
            self.weight_nums, self.bias_nums = weight_nums, bias_nums

            self.controller = nn.Linear(clip_dim, sum(weight_nums + bias_nums))
            self.forward_output = self.forward_output_entities

    def forward_encoder(self, x_in, normalize=True, metadata=None):
        # Backbone
        hidden_states_out, context = self.swinViT(x_in, normalize, metadata=metadata)
        context = context.permute(0, 2, 1)[:, :, :, None, None]

        # Skip connections
        enc0 = self.encoder1(x_in)  # [B, 48, 96, 96, 96]
        enc1 = self.encoder2(hidden_states_out[0])  # [B, 48, 48, 48, 48]
        enc2 = self.encoder3(hidden_states_out[1])  # [B, 96, 24, 24, 24]
        enc3 = self.encoder4(hidden_states_out[2])  # [B, 192, 12, 12, 12]
        enc4 = hidden_states_out[3]  # [B, 384, 6, 6, 6]

        # Bottleneck
        dec4 = self.encoder10(hidden_states_out[4])  # [B, 768, 3, 3, 3]
        return (enc0, enc1, enc2, enc3, enc4), (dec4, context)

    def forward_decoder(self, enc, dec4):
        dec3 = self.decoder5(dec4, enc[4])  # [B, 384, 6, 6, 6]
        dec2 = self.decoder4(dec3, enc[3])  # [B, 192, 12, 12, 12]
        dec1 = self.decoder3(dec2, enc[2])  # [B, 96, 24, 24, 24]
        dec0 = self.decoder2(dec1, enc[1])  # [B, 48, 48, 48, 48]
        out = self.decoder1(dec0, enc[0])  # [B, 48, 96, 96, 96]
        return out

    def forward_output_semseg(
        self, out_dec, weight_splits=None, bias_splits=None, num_insts=None
    ):
        out = self.output(out_dec)  # [B, CLASSES, 96, 96, 96]
        return out

    def get_dynamic_params(self, task_embedding):
        num_layers = len(self.weight_nums)
        params = self.controller(task_embedding)
        params = list(
            torch.split_with_sizes(params, self.weight_nums + self.bias_nums, dim=1)
        )

        # reshape the weights and biases to use them in conv layers
        weight_splits = params[:num_layers]
        bias_splits = params[num_layers:]
        return weight_splits, bias_splits

    def forward_output_entities(self, out_dec, weight_splits, bias_splits, num_insts):
        bs, _, H, W, D = out_dec.shape
        num_layers = len(self.weight_nums)

        # predict each element of the batch independently
        logits = []
        out_dec = self.precls_conv(out_dec)  # B, Ch=8, 96, 96, 96

        for B in range(len(out_dec)):
            x = out_dec[B].unsqueeze(0).repeat(num_insts, 1, 1, 1, 1)
            x = x.reshape(1, -1, H, W, D)  # 1, Cl * Ch, H, W, D
            for L in range(num_layers):
                weight_l = weight_splits[L].reshape(
                    num_insts * self.bias_nums[L], -1, 1, 1, 1
                )
                bias_l = bias_splits[L].reshape(num_insts * self.bias_nums[L])

                x = F.conv3d(
                    x,
                    weight_l,
                    bias=bias_l,
                    stride=1,
                    padding=0,
                    groups=num_insts,  # process every class independently (?)
                )
                x = F.relu(x)

            # return the logits
            x = self.forward_output_semseg(x.view(-1, self.bias_nums[-1], H, W, D))
            logits.append(x.squeeze(1))
        return logits

    def forward(self, x_in, normalize=True, metadata=None):
        # x_in: [B, 1, 96, 96, 96]
        # task_embedding: [C_task, clip_dim] x tasks, clip_dim=256

        # Encoder
        enc, (dec4, context) = self.forward_encoder(
            x_in, normalize=normalize, metadata=metadata
        )

        # Decoder
        out_dec = self.forward_decoder(enc, dec4)
        return out_dec
