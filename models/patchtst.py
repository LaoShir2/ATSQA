"""A compact PatchTST forecaster.

The interface follows the common Time-Series-Library forecasting shape:
input  x_enc: [batch, seq_len, channels]
output y_hat: [batch, pred_len, channels]
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.patch_num * d_model, pred_len)

        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x_enc):
        if x_enc.dim() != 3:
            raise ValueError("x_enc must have shape [batch, seq_len, channels]")

        # RevIN-style instance normalization without learnable affine params.
        mean = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True).clamp_min(1e-5)
        x = (x_enc - mean) / std

        batch, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.enc_in:
            raise ValueError(
                "expected [batch, %d, %d], got [batch, %d, %d]"
                % (self.seq_len, self.enc_in, seq_len, channels)
            )

        # [B, L, C] -> [B, C, patch_num, patch_len]
        patches = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(batch * channels, self.patch_num, self.patch_len)

        z = self.patch_proj(patches) + self.position
        z = self.encoder(self.dropout(z))
        z = z.reshape(batch * channels, self.patch_num * z.size(-1))
        out = self.head(z)

        # [B*C, pred_len] -> [B, pred_len, C]
        out = out.view(batch, channels, self.pred_len).permute(0, 2, 1)
        return out * std[:, -1:, :] + mean[:, -1:, :]
