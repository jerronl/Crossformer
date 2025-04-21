import torch, math
import torch.nn as nn
from einops import rearrange


def dsw(seg_len, linear, x, x2=None):
    batch, ts_len, ts_dim = x.shape
    pad_len = (-ts_len) % seg_len
    if pad_len:
        pad = x[:, -1:, :].expand(batch, pad_len, ts_dim)
        x = torch.cat([x, pad], dim=1)
        ts_len = ts_len + pad_len
    # seg_num = ts_len // seg_len
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
    x_embed = rearrange(
        x_embed, "(b d seg_num) d_model -> b d seg_num d_model", b=batch, d=ts_dim
    )
    return x_embed[:, : (ts_len - pad_len), :]


class MultiDSWEmbedding(nn.Module):
    def __init__(self, seg_lens, d_model, ycat, sect, sp):
        super().__init__()
        if not isinstance(seg_lens, list):
            seg_lens = [3, 11, 7, 5]

        self.embs = nn.ModuleList(
            [DSW_embedding(sl, d_model, ycat, sect, sp) for sl in seg_lens]
        )
        self.project = nn.LazyLinear(out_features=seg_lens[-1])

    def forward(self, x):
        embs = [emb(x) for emb in self.embs]
        return rearrange(
            self.project(rearrange(torch.cat(embs, dim=-2), "b t s d -> b t d s")),
            "b t d s -> b t s d",
        )


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model, ycat, sect, sp):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.use_prc = ycat < 1
        self.sp_shape = (sect, sp)

        self.sp_embed = nn.Sequential(
            nn.Unflatten(-1, self.sp_shape),
            nn.Linear(sp, sp * 2),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
            nn.Linear(sp * 2 * sect, sp * 2 + 2),
            nn.ReLU(),
        )

        self.linear1 = nn.Linear(seg_len, d_model)
        self.linear2 = nn.Linear(seg_len + 1, d_model)
        self.linear3 = nn.Linear(seg_len + 1, d_model) if self.use_prc else self.linear1

    def forward(self, x):
        xnp, xsp, cyclic, xpc, xvs, xvsp = x

        xsp = self.sp_embed(xsp)

        x_embed1 = dsw(self.seg_len, self.linear1, torch.cat((xnp, xsp), axis=2))
        x_embed2 = dsw(self.seg_len, self.linear2, cyclic, xvs)
        x_embed3 = dsw(self.seg_len, self.linear3, xpc, xvsp if self.use_prc else None)
        return torch.cat((x_embed1, x_embed2, x_embed3), 1)
