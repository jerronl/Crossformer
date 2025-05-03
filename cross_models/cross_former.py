import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from cross_models.cross_embed import MultiDSWEmbedding

from math import ceil


class Crossformer(nn.Module):
    def __init__(
        self,
        data_dim,
        out_dim,
        ycat,
        in_len,
        out_len,
        seg_len,
        sect,
        sp,
        win_size=4,
        factor=10,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        e_layers=3,
        dropout=0.0,
        baseline=False,
        device=torch.device("cuda:0"),
        x=None,
    ):
        super(Crossformer, self).__init__()
        self.device = device
        self.seg_len = seg_len
        self.in_len = in_len
        self.out_len = out_len

        # 1) Encoder‐side embedding
        self.enc_value_embedding = MultiDSWEmbedding(seg_len, d_model, ycat, sect, sp)

        # 2) Lazy inference of encoder channels & segment count
        with torch.no_grad():
            assert x is not None, "Must pass a dummy x for shape init"
            # x: [1, data_dim, in_len]
            out = self.enc_value_embedding(x)
            # out.shape = [1, value_dim, enc_seg_num, d_model]
            value_dim, enc_seg_num, _ = out.shape[1:]

        # 3) Encoder positional embedding
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, value_dim, enc_seg_num, d_model, device=device)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # 4) Encoder itself
        self.encoder = Encoder(
            e_layers,
            win_size,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            channels=value_dim,
            in_seg_num=enc_seg_num,
            factor=factor,
        )

        # 5) Decoder‐side positional embedding
        dec_seg_num = ceil(out_len / seg_len)
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, value_dim, dec_seg_num, d_model, device=device)
        )

        # 6) Decoder
        self.decoder = Decoder(
            seg_len,
            e_layers + 1,
            d_model,
            n_heads,
            d_ff,
            dropout,
            out_seg_num=dec_seg_num,
            factor=factor,
        )

        # 7) Final adapter: map from embedding channels → out_dim
        # self.adapter = nn.Sequential(
        #     nn.Linear(value_dim, out_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(out_dim * 2, out_dim),
        # )
        self.adapter_mu = nn.Sequential(
            nn.Linear(value_dim, out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim)
        )
        self.adapter_q90 = nn.Sequential(
            nn.Linear(value_dim, out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim)
        )
        self.baseline = baseline

    def forward(self, x_seq):
        """
        x_seq: [batch, data_dim, in_len]
        returns: [batch, out_len, out_dim]
        """
        batch = x_seq[0].shape[0]

        # (optional) baseline
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)  # [B,1,in_len]
        else:
            base = 0

        # 1) pad to multiple of seg_len if needed
        pad_in = ceil(self.in_len / self.seg_len) * self.seg_len - self.in_len
        if pad_in > 0:
            # replicate first step to the left
            left = x_seq[:, :1, :].expand(batch, -1, pad_in)
            x_seq = torch.cat([left, x_seq], dim=2)

        # 2) Encoder embedding + pos‑emb + norm
        x_enc = self.enc_value_embedding(x_seq)  # [B, C, S, d_model]
        x_enc = x_enc + self.enc_pos_embedding  # broadcast on B
        x_enc = self.pre_norm(x_enc)  # LayerNorm on d_model

        # 3) Encode
        enc_out = self.encoder(x_enc)

        # 4) Prepare decoder input: just the pos‑emb repeated to B
        # dec_in = self.dec_pos_embedding.repeat(batch, 1, 1, 1)  # [B, C, S_dec, d_model]

        # # 5) Decode
        # dec_out = self.decoder(dec_in, enc_out)  # [B, S_dec, C]

        # # 6) Adapt to final output dimension
        # #    decoder returns [B, S_dec, C], adapter expects (B*S_dec, C)
        # #    but Linear on last dim works with [B, S_dec, C] directly in PyTorch
        # predict = self.adapter(dec_out)  # [B, S_dec, out_dim]
        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=batch,
        )        

        # 5) Decode
        dec_out = self.decoder(dec_in, enc_out)  # [B, S_dec, C]

        # 6) Adapt to final output dimension
        #    decoder returns [B, S_dec, C], adapter expects (B*S_dec, C)
        #    but Linear on last dim works with [B, S_dec, C] directly in PyTorch
        # predict = self.adapter(dec_out)  # [B, S_dec, out_dim]

        # 7) slice to original horizon
        # predict = predict[:, : self.out_len, :]

        # return base + predict
        pred_mu  = self.adapter_mu(dec_out)    # [B, pad_seg_num, out_dim]
        pred_q90 = self.adapter_q90(dec_out)

        pred_mu  = base + pred_mu[:, : self.out_len, :]
        pred_q90 = base + pred_q90[:, : self.out_len, :]
        
        return pred_mu, pred_q90