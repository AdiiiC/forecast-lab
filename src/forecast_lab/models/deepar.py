"""DeepAR: autoregressive RNN with Gaussian / Student-t likelihood head.

Trains by teacher-forcing on rolling windows; forecasts by sampling S trajectories.
Lightweight self-contained PyTorch implementation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ..distributions import Empirical
from .base import BaseModel, Forecast


class _DeepARNet(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.1, dist="studentt"):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.dist = dist
        self.mu_head    = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)
        self.nu_head    = nn.Linear(hidden, 1) if dist == "studentt" else None

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        mu = self.mu_head(h).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.sigma_head(h)).squeeze(-1) + 1e-3
        if self.nu_head is not None:
            nu = 2.0 + torch.nn.functional.softplus(self.nu_head(h)).squeeze(-1)
            return mu, sigma, nu, state
        return mu, sigma, None, state


def _nll(y, mu, sigma, nu, dist):
    if dist == "gaussian":
        return 0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)
    # student-t
    z = (y - mu) / sigma
    return (torch.lgamma(nu / 2) - torch.lgamma((nu + 1) / 2)
            + 0.5 * torch.log(nu * np.pi) + torch.log(sigma)
            + (nu + 1) / 2 * torch.log1p(z * z / nu))


class DeepARModel(BaseModel):
    name = "deepar"
    produces_intervals = True
    produces_distribution = True

    def __init__(self, context: int = 168, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, dist: str = "studentt",
                 epochs: int = 20, batch_size: int = 128, lr: float = 1e-3,
                 n_samples: int = 200):
        self.context, self.hidden, self.layers, self.dropout = context, hidden, layers, dropout
        self.dist, self.epochs, self.bs, self.lr = dist, epochs, batch_size, lr
        self.n_samples = n_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _windows(self, y):
        L = self.context
        n = len(y) - L - 1
        X = np.lib.stride_tricks.sliding_window_view(y, L)[:n].astype(np.float32)
        Y = np.lib.stride_tricks.sliding_window_view(y[1:], L)[:n].astype(np.float32)
        return X[..., None], Y  # input has channel dim

    def fit(self, y: pd.Series):
        self.y_ = y.copy()
        self.mu0_, self.sd0_ = float(y.mean()), float(y.std() + 1e-8)
        ys = (y.values - self.mu0_) / self.sd0_
        X, Y = self._windows(ys)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=self.bs, shuffle=True, drop_last=True)
        self.net_ = _DeepARNet(self.hidden, self.layers, self.dropout, self.dist).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        self.net_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                mu, sigma, nu, _ = self.net_(xb)
                loss = _nll(yb, mu, sigma, nu, self.dist).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, horizon: int, alpha: float = 0.1) -> Forecast:
        ys = (self.y_.values - self.mu0_) / self.sd0_
        ctx = torch.from_numpy(ys[-self.context:].astype(np.float32))[None, :, None].to(self.device)
        # Warm-up state on the context
        self.net_.eval()
        _, _, _, state = self.net_(ctx)
        last = ctx[:, -1:, :].expand(self.n_samples, -1, -1).clone()
        # Tile state across samples
        h, c = state
        state = (h.repeat(1, self.n_samples, 1), c.repeat(1, self.n_samples, 1))
        samples = np.empty((horizon, self.n_samples), dtype=np.float32)
        for t in range(horizon):
            mu, sigma, nu, state = self.net_(last, state)
            mu, sigma = mu[:, -1], sigma[:, -1]
            if self.dist == "gaussian":
                eps = torch.randn_like(mu)
                draw = mu + sigma * eps
            else:
                nu = nu[:, -1]
                # t-sample via normal/chi2
                z = torch.randn_like(mu)
                g = torch.distributions.Chi2(nu).sample() / nu
                draw = mu + sigma * z / torch.sqrt(g)
            samples[t] = draw.cpu().numpy()
            last = draw.view(self.n_samples, 1, 1)
        samples = samples * self.sd0_ + self.mu0_
        dist = Empirical(samples=samples)
        lo, hi = dist.interval(alpha)
        return Forecast(mean=dist.mean(), lo=lo, hi=hi,
                        samples=samples, dist=dist)