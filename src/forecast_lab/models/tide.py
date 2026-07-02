"""TiDE: Time-series Dense Encoder.

A pure-MLP long-horizon forecaster (Das et al. 2023) — surprisingly competitive
vs. Transformers and an order of magnitude faster. We add a known-future
covariate stream when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, Forecast
from ..calendars import calendar_known_future


class _ResBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

        self.drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_out)

        self.skip = (
            nn.Linear(d_in, d_out)
            if d_in != d_out
            else nn.Identity()
        )

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)

        h = self.fc2(h)
        h = self.drop(h)

        return self.norm(h + self.skip(x))


class _TiDENet(nn.Module):
    def __init__(
        self,
        context,
        horizon,
        n_fut_feats,
        hidden=256,
        layers=2,
        dropout=0.1,
    ):
        super().__init__()

        d_in = context + horizon * n_fut_feats

        self.enc = nn.Sequential(
            *[
                _ResBlock(
                    d_in if i == 0 else hidden,
                    hidden,
                    hidden,
                    dropout,
                )
                for i in range(layers)
            ]
        )

        self.dec = nn.Sequential(
            *[
                _ResBlock(
                    hidden,
                    hidden,
                    hidden,
                    dropout,
                )
                for _ in range(layers)
            ]
        )

        self.head = nn.Linear(hidden, horizon)

        # Global linear skip connection
        self.skip = nn.Linear(context, horizon)

    def forward(self, x_past, x_fut):
        z = torch.cat(
            [x_past, x_fut.flatten(1)],
            dim=1,
        )

        z = self.enc(z)
        z = self.dec(z)

        return self.head(z) + self.skip(x_past)


class TiDEModel(BaseModel):
    name = "tide"

    # Wrap with conformal externally for intervals
    produces_intervals = False

    accepts_covariates = True

    def __init__(
        self,
        context: int = 336,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        country: str | None = None,
    ):
        self.context = context
        self.hidden = hidden
        self.layers = layers

        self.dropout = dropout
        self.epochs = epochs
        self.bs = batch_size
        self.lr = lr

        self.country = country

        from ..device import pick_device
        self.device = pick_device()

    def _future_features(self, idx):
        cal = calendar_known_future(
            idx,
            country=self.country,
        ).values.astype(np.float32)

        return cal

    def _windows(self, y, idx, horizon):
        L = self.context

        n = len(y) - L - horizon + 1

        if n <= 0:
            raise ValueError(
                f"Not enough history for context={L} "
                f"and horizon={horizon}"
            )

        # Past windows
        X_past = np.lib.stride_tricks.sliding_window_view(
            y,
            L,
        )[:n].astype(np.float32)

        # Targets
        Y = np.lib.stride_tricks.sliding_window_view(
            y[L:],
            horizon,
        )[:n].astype(np.float32)

        # Future known covariates
        fut = self._future_features(idx)

        X_fut = np.lib.stride_tricks.sliding_window_view(
            fut[L:],
            (horizon, fut.shape[1]),
        )

        X_fut = X_fut[:n, 0].astype(np.float32)

        return X_past, X_fut, Y

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()

        self.mu0_ = float(y.mean())
        self.sd0_ = float(y.std() + 1e-8)

        return self

    def _train(self, horizon):
        ys = (
            self.y_.values - self.mu0_
        ) / self.sd0_

        Xp, Xf, Y = self._windows(
            ys,
            self.y_.index,
            horizon,
        )

        self.n_fut_ = Xf.shape[-1]

        ds = TensorDataset(
            torch.from_numpy(Xp),
            torch.from_numpy(Xf),
            torch.from_numpy(Y),
        )

        dl = DataLoader(
            ds,
            batch_size=self.bs,
            shuffle=True,
            drop_last=True,
        )

        net = _TiDENet(
            context=self.context,
            horizon=horizon,
            n_fut_feats=self.n_fut_,
            hidden=self.hidden,
            layers=self.layers,
            dropout=self.dropout,
        ).to(self.device)

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        loss_fn = nn.SmoothL1Loss()

        net.train()

        for epoch in range(self.epochs):

            epoch_loss = 0.0

            for xp, xf, yb in dl:

                xp = xp.to(self.device)
                xf = xf.to(self.device)
                yb = yb.to(self.device)

                opt.zero_grad()

                pred = net(xp, xf)

                loss = loss_fn(pred, yb)

                # DEBUG SAFETY CHECK
                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss tensor is detached from graph."
                    )

                loss.backward()

                opt.step()

                epoch_loss += loss.item()

        self.net_ = net
        self.horizon_ = horizon

    def predict(
        self,
        horizon: int,
        alpha: float = 0.1,
        cov=None,
    ) -> Forecast:

        # Train lazily if needed
        if (
            not hasattr(self, "net_")
            or self.horizon_ != horizon
        ):
            self._train(horizon)

        ys = (
            self.y_.values - self.mu0_
        ) / self.sd0_

        freq = pd.infer_freq(self.y_.index) or "H"

        fut_idx = pd.date_range(
            self.y_.index[-1]
            + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

        xp = torch.from_numpy(
            ys[-self.context:].astype(np.float32)
        )[None].to(self.device)

        xf = torch.from_numpy(
            self._future_features(fut_idx)
        )[None].to(self.device)

        self.net_.eval()

        # IMPORTANT:
        # no_grad ONLY during inference
        with torch.no_grad():

            out = self.net_(xp, xf)

            out = (
                out.cpu().numpy()[0]
                * self.sd0_
                + self.mu0_
            )

        return Forecast(mean=out)