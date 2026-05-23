"""Compact Temporal Fusion Transformer with quantile output head.

Gated residual networks + multi-head attention + horizon × K quantile tensor
trained with pinball loss.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ..distributions import Quantile
from .base import BaseModel, Forecast


class _GRN(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d); self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d); self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = torch.nn.functional.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(h))
        return self.norm(x + g * h)


class _TFTNet(nn.Module):
    def __init__(self, context, horizon, d=64, heads=4, dropout=0.1, q_levels=None):
        super().__init__()
        self.context, self.horizon = context, horizon
        self.q_levels = q_levels
        self.embed = nn.Linear(1, d)
        self.lstm  = nn.LSTM(d, d, batch_first=True)
        self.grn1  = _GRN(d, dropout)
        self.attn  = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.grn2  = _GRN(d, dropout)
        self.head  = nn.Linear(d, horizon * len(q_levels))

    def forward(self, x):
        z = self.embed(x)
        z, _ = self.lstm(z)
        z = self.grn1(z)
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.grn2(z + a)
        z = z[:, -1, :]
        out = self.head(z).view(x.size(0), self.horizon, len(self.q_levels))
        return out


def _pinball(y, q_hat, q_levels):
    # y: (B, H), q_hat: (B, H, K)
    y = y.unsqueeze(-1)
    q = torch.tensor(q_levels, dtype=torch.float32, device=y.device).view(1, 1, -1)
    diff = y - q_hat
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))


class TFTModel(BaseModel):
    name = "tft"
    produces_intervals = True
    produces_distribution = True
    accepts_covariates = False

    def __init__(self, context: int = 168, horizon_max: int = 24,
                 d_model: int = 64, heads: int = 4, dropout: float = 0.1,
                 epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
                 q_levels=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
        self.context, self.horizon_max = context, horizon_max
        self.d, self.heads, self.dropout = d_model, heads, dropout
        self.epochs, self.bs, self.lr = epochs, batch_size, lr
        self.q_levels = tuple(q_levels)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _windows(self, y, horizon):
        L = self.context
        n = len(y) - L - horizon + 1
        X = np.lib.stride_tricks.sliding_window_view(y, L)[:n].astype(np.float32)
        Y = np.lib.stride_tricks.sliding_window_view(y[L:], horizon)[:n].astype(np.float32)
        return X[..., None], Y

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()
        self.mu0_, self.sd0_ = float(y.mean()), float(y.std() + 1e-8)
        return self

    def _train(self, horizon: int):
        ys = (self.y_.values - self.mu0_) / self.sd0_
        X, Y = self._windows(ys, horizon)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=self.bs, shuffle=True, drop_last=True)
        net = _TFTNet(self.context, horizon, self.d, self.heads,
                      self.dropout, list(self.q_levels)).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                qhat = net(xb)
                loss = _pinball(yb, qhat, self.q_levels)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net_ = net
        self.horizon_ = horizon

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        # Train lazily (must happen OUTSIDE no_grad)
        if not hasattr(self, "net_") or self.horizon_ != horizon:
            self._train(horizon)

        # Inference only — gradients disabled here
        ys = (self.y_.values - self.mu0_) / self.sd0_
        x = torch.from_numpy(
            ys[-self.context:].astype(np.float32)
        )[None, :, None].to(self.device)

        self.net_.eval()
        with torch.no_grad():
            q_hat = self.net_(x).cpu().numpy()[0]   # (H, K)

        q_hat = q_hat * self.sd0_ + self.mu0_
        q_levels = np.array(self.q_levels)
        dist = Quantile(q_levels=q_levels, q_values=q_hat)
        mean = q_hat[:, np.argmin(np.abs(q_levels - 0.5))]
        lo = q_hat[:, np.argmin(np.abs(q_levels - alpha / 2))]
        hi = q_hat[:, np.argmin(np.abs(q_levels - (1 - alpha / 2)))]
        return Forecast(mean=mean, lo=lo, hi=hi,
                        quantiles=q_hat, q_levels=q_levels, dist=dist)