"""PatchTST: patch-wise Transformer for long-horizon forecasting.

Following Nie et al. 2023: split the look-back into non-overlapping patches,
embed each patch, run a vanilla Transformer encoder over the patch tokens,
then a linear head maps the flattened token output to horizon × quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..distributions import Quantile
from .base import BaseModel, Forecast


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        context,
        horizon,
        patch_len,
        stride,
        d_model,
        heads,
        layers,
        dropout,
        n_quantiles,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        self.n_patches = (context - patch_len) // stride + 1

        self.proj = nn.Linear(patch_len, d_model)

        self.pos = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model)
        )

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            enc,
            num_layers=layers,
        )

        self.head = nn.Linear(
            self.n_patches * d_model,
            horizon * n_quantiles,
        )

        self.horizon = horizon
        self.n_quantiles = n_quantiles

    def _patchify(self, x):
        """
        x: (B, L)
        returns: (B, n_patches, patch_len)
        """
        return x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )

    def forward(self, x):
        p = self._patchify(x)

        z = self.proj(p)
        z = z + self.pos

        z = self.encoder(z)

        z = z.flatten(1)

        out = self.head(z)

        return out.view(
            x.size(0),
            self.horizon,
            self.n_quantiles,
        )


def _pinball(y, q_hat, q_levels):
    """
    Quantile / pinball loss.

    y:      (B, H)
    q_hat:  (B, H, Q)
    """

    y = y.unsqueeze(-1)

    q = torch.tensor(
        q_levels,
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, -1)

    diff = y - q_hat

    loss = torch.maximum(
        q * diff,
        (q - 1.0) * diff,
    )

    return loss.mean()


class PatchTSTModel(BaseModel):
    name = "patchtst"

    produces_intervals = True
    produces_distribution = True

    def __init__(
        self,
        context: int = 336,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-3,
        q_levels=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    ):
        self.context = context
        self.patch_len = patch_len
        self.stride = stride

        self.d = d_model
        self.heads = heads
        self.layers = layers
        self.dropout = dropout

        self.epochs = epochs
        self.bs = batch_size
        self.lr = lr

        self.q_levels = tuple(q_levels)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def _windows(self, y, horizon):
        L = self.context

        n = len(y) - L - horizon + 1

        if n <= 0:
            raise ValueError(
                f"Not enough history for context={L} "
                f"and horizon={horizon}"
            )

        X = np.lib.stride_tricks.sliding_window_view(
            y,
            L,
        )[:n].astype(np.float32)

        Y = np.lib.stride_tricks.sliding_window_view(
            y[L:],
            horizon,
        )[:n].astype(np.float32)

        return X, Y

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()

        self.mu0_ = float(y.mean())
        self.sd0_ = float(y.std() + 1e-8)

        return self

    def _train(self, horizon):
        # ensure gradients are enabled
        torch.set_grad_enabled(True)

        ys = (
            (self.y_.values - self.mu0_)
            / self.sd0_
        )

        X, Y = self._windows(ys, horizon)

        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y),
        )

        dl = DataLoader(
            ds,
            batch_size=self.bs,
            shuffle=True,
            drop_last=True,
        )

        net = _PatchTSTNet(
            context=self.context,
            horizon=horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d,
            heads=self.heads,
            layers=self.layers,
            dropout=self.dropout,
            n_quantiles=len(self.q_levels),
        ).to(self.device)

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        net.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                opt.zero_grad()

                q_hat = net(xb)

                loss = _pinball(
                    yb,
                    q_hat,
                    self.q_levels,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=1.0,
                )

                opt.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dl), 1)

            print(
                f"[PatchTST] "
                f"epoch={epoch + 1}/{self.epochs} "
                f"loss={avg_loss:.6f}"
            )

        self.net_ = net
        self.horizon_ = horizon

    def predict(
        self,
        horizon: int,
        alpha: float = 0.1,
        cov=None,
    ) -> Forecast:

        # retrain if needed
        if (
            not hasattr(self, "net_")
            or self.horizon_ != horizon
        ):
            self._train(horizon)

        ys = (
            (self.y_.values - self.mu0_)
            / self.sd0_
        )

        x = torch.from_numpy(
            ys[-self.context:].astype(np.float32)
        )[None].to(self.device)

        self.net_.eval()

        # inference only
        with torch.no_grad():
            q_hat = self.net_(x)

        q_hat = (
            q_hat.cpu().numpy()[0]
            * self.sd0_
            + self.mu0_
        )

        q_levels = np.array(self.q_levels)

        median_idx = np.argmin(
            np.abs(q_levels - 0.5)
        )

        lo_idx = np.argmin(
            np.abs(q_levels - alpha / 2)
        )

        hi_idx = np.argmin(
            np.abs(q_levels - (1 - alpha / 2))
        )

        return Forecast(
            mean=q_hat[:, median_idx],
            lo=q_hat[:, lo_idx],
            hi=q_hat[:, hi_idx],
            quantiles=q_hat,
            q_levels=q_levels,
            dist=Quantile(
                q_levels=q_levels,
                q_values=q_hat,
            ),
        )