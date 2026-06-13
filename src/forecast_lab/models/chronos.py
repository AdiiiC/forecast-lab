"""Zero-shot foundation model adapter for the Chronos family (v2.2+).

Supports three model types:
  * Chronos-2        (amazon/chronos-2) — newest, supports covariates natively
  * Chronos-Bolt     (amazon/chronos-bolt-*) — fast quantile-direct models
  * Chronos-T5       (amazon/chronos-t5-*) — original sample-based models

Install: pip install chronos-forecasting>=2.2
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..distributions import Empirical, Quantile
from .base import BaseModel, Forecast


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class ChronosModel(BaseModel):
    name = "chronos"
    produces_intervals = True
    produces_distribution = True
    accepts_covariates = True   # Chronos-2 supports covariates natively

    def __init__(self, model_id: str = "amazon/chronos-bolt-small",
                 num_samples: int = 200,
                 q_levels: tuple = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
                 device: str | None = None):
        self.model_id = model_id
        self.num_samples = num_samples
        self.q_levels = list(q_levels)
        self.device = device or ("cuda" if _has_cuda() else "cpu")

    def _is_chronos2(self) -> bool:
        return "chronos-2" in self.model_id

    def _is_bolt(self) -> bool:
        return "chronos-bolt" in self.model_id

    def _load(self):
        # Try loading onto the requested device; if that fails (e.g. due to
        # meta-tensor/device dispatch errors from transformers/accelerate),
        # retry on CPU to provide a safe fallback.
        last_err = None
        for target_device in (self.device, "cpu"):
            try:
                if self._is_chronos2():
                    from chronos import Chronos2Pipeline  # type: ignore
                    self.pipe_ = Chronos2Pipeline.from_pretrained(
                        self.model_id, device_map=target_device)
                    self.kind_ = "chronos2"
                else:
                    from chronos import ChronosPipeline  # type: ignore
                    self.pipe_ = ChronosPipeline.from_pretrained(
                        self.model_id, device_map=target_device)
                    self.kind_ = "bolt" if self._is_bolt() else "t5"
                # Record the actual device used
                self.device = target_device
                return
            except Exception as e:
                last_err = e
                # If already tried CPU, break and raise below
                if target_device == "cpu":
                    break
                # Otherwise, log and retry with CPU
        raise RuntimeError(
            "Install chronos-forecasting>=2.2: `pip install chronos-forecasting`. "
            f"Failed to load Chronos model '{self.model_id}'. Last error: {last_err}")

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()
        self.cov_ = cov
        if not hasattr(self, "pipe_"):
            self._load()
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        if self.kind_ == "chronos2":
            return self._predict_chronos2(horizon, alpha, cov)
        elif self.kind_ == "bolt":
            return self._predict_bolt(horizon, alpha)
        else:
            return self._predict_t5(horizon, alpha)

    def _predict_chronos2(self, horizon: int, alpha: float, cov) -> Forecast:
        """Chronos-2: uses predict_df with DataFrame interface."""
        # Build context dataframe
        context_df = pd.DataFrame({
            "timestamp": self.y_.index,
            "target": self.y_.values,
            "id": "series_0",
        })

        # Build future covariates if available
        freq = pd.infer_freq(self.y_.index) or "H"
        future_idx = pd.date_range(
            self.y_.index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, freq=freq)
        future_df = None
        real_cov = cov or self.cov_
        if real_cov is not None and hasattr(real_cov, "known_future") and not real_cov.known_future.empty:
            future_df = real_cov.known_future.reindex(future_idx).reset_index()
            future_df.columns = ["timestamp"] + list(real_cov.known_future.columns)
            future_df["id"] = "series_0"

        pred_df = self.pipe_.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=self.q_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # Extract quantiles from result columns
        q_levels = np.array(self.q_levels)
        q_values = np.column_stack([
            pred_df[str(q)].values if str(q) in pred_df.columns
            else pred_df[f"{q:.2f}".rstrip('0').rstrip('.')].values
            for q in self.q_levels
        ]).astype(float)

        # Point forecast = median
        med_idx = np.argmin(np.abs(q_levels - 0.5))
        if "predictions" in pred_df.columns:
            mean = pred_df["predictions"].values.astype(float)
        else:
            mean = q_values[:, med_idx]

        lo_idx = np.argmin(np.abs(q_levels - alpha / 2))
        hi_idx = np.argmin(np.abs(q_levels - (1 - alpha / 2)))
        dist = Quantile(q_levels=q_levels, q_values=q_values)

        return Forecast(
            mean=mean, lo=q_values[:, lo_idx], hi=q_values[:, hi_idx],
            quantiles=q_values, q_levels=q_levels, dist=dist,
        )

    def _predict_bolt(self, horizon: int, alpha: float) -> Forecast:
        """Chronos-Bolt: quantile-direct forecasting via predict_quantiles."""
        import torch
        ctx = torch.tensor(self.y_.values, dtype=torch.float32)
        q_levels = np.array(self.q_levels)

        # Bolt models support predict_quantiles
        quantiles, mean_pred = self.pipe_.predict_quantiles(
            context=ctx,
            prediction_length=horizon,
            quantile_levels=self.q_levels,
        )
        # quantiles shape: (1, H, K), mean_pred shape: (1, H)
        q_values = quantiles[0].cpu().numpy().astype(float)   # (H, K)
        mean = mean_pred[0].cpu().numpy().astype(float)       # (H,)

        lo_idx = np.argmin(np.abs(q_levels - alpha / 2))
        hi_idx = np.argmin(np.abs(q_levels - (1 - alpha / 2)))
        dist = Quantile(q_levels=q_levels, q_values=q_values)

        return Forecast(
            mean=mean, lo=q_values[:, lo_idx], hi=q_values[:, hi_idx],
            quantiles=q_values, q_levels=q_levels, dist=dist,
        )

    def _predict_t5(self, horizon: int, alpha: float) -> Forecast:
        """Original Chronos-T5: sample-based forecasting via predict."""
        import torch
        ctx = torch.tensor(self.y_.values, dtype=torch.float32)

        # T5 models return sample trajectories: (1, S, H)
        samples = self.pipe_.predict(
            ctx,
            prediction_length=horizon,
            num_samples=self.num_samples,
        ).cpu().numpy()[0].T          # → (H, S)

        dist = Empirical(samples=samples)
        lo, hi = dist.interval(alpha)
        return Forecast(
            mean=dist.mean(), lo=lo, hi=hi,
            samples=samples, dist=dist,
        ) 