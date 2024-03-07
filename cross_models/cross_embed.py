import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

import math


def dsw(seg_len, linear, x, x2=None):
    batch, ts_len, ts_dim = x.shape
    x_segment = rearrange(
        x, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=seg_len
    )
    if x2 is not None:
        assert x2.shape[0] == x.shape[0] and x2.ndim == 2
        x2_segment = x2.repeat_interleave(
            ts_len // seg_len * ts_dim // x2.shape[1], dim=1
        ).reshape(-1, 1)
        x_segment = torch.cat((x_segment, x2_segment), axis=1)
    x_embed = linear(x_segment)
    return rearrange(
        x_embed, "(b d seg_num) d_model -> b d seg_num d_model", b=batch, d=ts_dim
    )


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model, ycat):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.use_prc = 1 if ycat>0 else 0

        self.linear1 = nn.Linear(seg_len, d_model)
        self.linear2 = nn.Linear(seg_len + 1, d_model)
        self.linear3 = nn.Linear(seg_len + self.use_prc, d_model)

    def forward(self, x):
        xnp, cyclic, xpc, xvs, xvsp = x
        x_embed1 = dsw(
            self.seg_len,
            self.linear1,
            xnp,
        )
        x_embed2 = dsw(self.seg_len, self.linear2, cyclic, xvs)
        x_embed3 = dsw(self.seg_len, self.linear3, xpc, xvsp if self.use_prc else None)
        return torch.cat((x_embed1, x_embed2, x_embed3), 1)
