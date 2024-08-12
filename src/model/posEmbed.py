import numpy as np

import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor, spacing=[[1, 1, 1]], upper_corner=[[0, 0, 0]]):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        # scale and shift based on the image spacing and crop coordinates
        sin_inp_x = (
            torch.einsum("b,i->bi", spacing[:, 0], pos_x) + upper_corner[:, 0][:, None]
        )  # b,x
        sin_inp_y = (
            torch.einsum("b,i->bi", spacing[:, 1], pos_y) + upper_corner[:, 1][:, None]
        )  # b,y
        sin_inp_z = (
            torch.einsum("b,i->bi", spacing[:, 2], pos_z) + upper_corner[:, 2][:, None]
        )  # b,z

        # sine and cosine signals
        sin_inp_x = torch.einsum("bi,j->bij", sin_inp_x, self.inv_freq)  # b,x,ch
        sin_inp_y = torch.einsum("bi,j->bij", sin_inp_y, self.inv_freq)  # b,y,ch
        sin_inp_z = torch.einsum("bi,j->bij", sin_inp_z, self.inv_freq)  # b,z,ch

        # combining everything
        emb_x = get_emb(sin_inp_x).unsqueeze(2).unsqueeze(2)  # b,x,1,1,ch
        emb_y = get_emb(sin_inp_y).unsqueeze(2).unsqueeze(1)  # b,1,y,1,ch
        emb_z = get_emb(sin_inp_z).unsqueeze(1).unsqueeze(1)  # b,1,1,z,ch
        emb = torch.zeros(
            (batch_size, x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, :, : self.channels] = emb_x
        emb[:, :, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[:, :, :, :, :orig_ch]
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor, spacing, center):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor, spacing, center)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels
