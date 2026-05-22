"""N-BEATS (generic stack) with MC-dropout epistemic prediction intervals.

Faithful to Oreshkin et al. (2019) — generic architecture with fully-connected
stacks, backcast/forecast factoring. MC-dropout at inference yields a spread of
trajectories whose quantiles form the PI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModel, Forecast


class _Block(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),     nn.ReLU(), nn.Dropout(dropout),
        )
        self.b = nn.Linear(hidden, input_size)   # backcast
        self.f = nn.Linear(hidden, horizon)      # forecast

    def forward(self, x):
        h = self.fc(x)
        return self.b(h), self.f(h)


class _NBeatsNet(nn.Module):
    def __init__(self, input_size: int, horizon: int, stacks: int = 3,
                 blocks: int = 3, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Block(input_size, horizon, hidden, dropout)
             for _ in range(stacks * blocks)]
        )

    def forward(self, x):
        residual = x
        forecast = 0.0
        for blk in self.blocks:
            b, f = blk(residual)
            residual = residual - b
            forecast = forecast + f
        return forecast


class NBeatsModel(BaseModel):
    name = "nbeats"
    produces_intervals = True
    accepts_covariates = False

    def __init__(self, input_size: int = 168, hidden: int = 256,
                 stacks: int = 3, blocks: int = 3,
                 epochs: int = 30, batch_size: int = 256,
                 dropout: float = 0.1, lr: float = 1e-3,
                 mc_samples: int = 50):
        self.input_size = input_size
        self.hidden = hidden
        self.stacks = stacks
        self.blocks = blocks
        self.epochs = epochs
        self.bs = batch_size
        self.dropout = dropout
        self.lr = lr
        self.mc_samples = mc_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _make_windows(self, y: np.ndarray, horizon: int):
        L = self.input_size
        n = len(y) - L - horizon + 1
        X = np.lib.stride_tricks.sliding_window_view(y, L)[:n]
        Y = np.lib.stride_tricks.sliding_window_view(y[L:], horizon)[:n]
        return X.astype(np.float32), Y.astype(np.float32)

    def fit(self, y: pd.Series, cov=None) -> "NBeatsModel":
        self.y_ = y.copy()
        self.mu_ = float(y.mean())
        self.sd_ = float(y.std() + 1e-8)
        return self  # actual training deferred to predict() since horizon is needed

    def _train(self, horizon: int):
        y = (self.y_.values - self.mu_) / self.sd_
        X, Y = self._make_windows(y, horizon)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=self.bs, shuffle=True, drop_last=True)

        net = _NBeatsNet(
            self.input_size, horizon,
            stacks=self.stacks, blocks=self.blocks,
            hidden=self.hidden, dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.SmoothL1Loss()

        net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
        self.net_ = net
        self.trained_horizon_ = horizon

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        # Retrain if first call or horizon changed
        if not hasattr(self, "net_") or self.trained_horizon_ != horizon:
            self._train(horizon)

        y = (self.y_.values - self.mu_) / self.sd_
        x = torch.from_numpy(
            y[-self.input_size:].astype(np.float32)
        )[None].to(self.device)

        # MC-dropout: keep dropout active at inference for epistemic uncertainty
        self.net_.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.net_(x).cpu().numpy()[0])
        preds = np.stack(preds) * self.sd_ + self.mu_

        mean = preds.mean(axis=0)
        lo = np.quantile(preds, alpha / 2, axis=0)
        hi = np.quantile(preds, 1 - alpha / 2, axis=0)
        return Forecast(mean=mean, lo=lo, hi=hi, samples=preds.T)