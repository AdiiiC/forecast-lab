"""Zero-shot foundation model adapter for the Chronos family (v1.4+).

Supports two model types:
  * Chronos-Bolt     (amazon/chronos-bolt-*) — fast quantile-direct models
  * Chronos-T5       (amazon/chronos-t5-*) — original sample-based models

Install: pip install chronos-forecasting>=1.4
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
    accepts_covariates = False

    def __init__(self, model_id: str = "amazon/chronos-bolt-small",
                 num_samples: int = 200,
                 q_levels: tuple = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
                 device: str | None = None):
        self.model_id = model_id
        self.num_samples = num_samples
        self.q_levels = list(q_levels)
        self.device = device or ("cuda" if _has_cuda() else "cpu")

    def _is_bolt(self) -> bool:
        return "chronos-bolt" in self.model_id

    def _load(self):
        import torch
        try:
            from chronos import ChronosPipeline  # type: ignore
            # low_cpu_mem_usage=False: disables meta-tensor initialisation which
            # leaves generation_config special-token tensors on meta device,
            # causing "Tensor.item() cannot be called on meta tensors" at
            # inference time (transformers >=4.40 + chronos-t5 models).
            # device_map on CPU also triggers accelerate's dispatch_model which
            # has the same meta-tensor issue, so we skip it for CPU.
            load_kwargs: dict = {"dtype": torch.float32, "low_cpu_mem_usage": False}
            if self.device != "cpu":
                load_kwargs["device_map"] = self.device
            self.pipe_ = ChronosPipeline.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
            self.kind_ = "bolt" if self._is_bolt() else "t5"
        except Exception as e:
            raise RuntimeError(
                "Install chronos-forecasting>=1.4: "
                "`pip install chronos-forecasting`. "
                f"Underlying error: {e}")

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()
        if not hasattr(self, "pipe_"):
            self._load()
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        if self.kind_ == "bolt":
            return self._predict_bolt(horizon, alpha)
        else:
            return self._predict_t5(horizon, alpha)

    def _predict_bolt(self, horizon: int, alpha: float) -> Forecast:
        """Chronos-Bolt: quantile-direct forecasting via predict_quantiles."""
        import torch
        ctx = torch.tensor(self.y_.values, dtype=torch.float32)
        q_levels = np.array(self.q_levels)

        # Bolt models support predict_quantiles — returns (B, H, K) tensor
        quantiles_tensor = self.pipe_.predict_quantiles(
            context=ctx,
            prediction_length=horizon,
            quantile_levels=self.q_levels,
        )
        # quantiles shape: (1, H, K)
        q_values = quantiles_tensor[0].detach().cpu().float().numpy()  # (H, K)

        # Use median as point forecast
        med_idx = np.argmin(np.abs(q_levels - 0.5))
        mean = q_values[:, med_idx]

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
        ).detach().cpu().float().numpy()[0].T   # → (H, S)

        dist = Empirical(samples=samples)
        lo, hi = dist.interval(alpha)
        return Forecast(
            mean=dist.mean(), lo=lo, hi=hi,
            samples=samples, dist=dist,
        ) 