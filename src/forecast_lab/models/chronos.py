"""Zero-shot foundation model adapter: Chronos (Amazon) / TimesFM (Google).

Both expose a `forecast(context, horizon, num_samples)` API via huggingface.
This adapter tries Chronos first, then TimesFM. If neither is installed it
raises a clear error at fit time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..distributions import Empirical
from .base import BaseModel, Forecast


class ChronosModel(BaseModel):
    name = "chronos"
    produces_intervals = True
    produces_distribution = True

    def __init__(self, model_id: str = "amazon/chronos-t5-small",
                 num_samples: int = 200, device: str | None = None,
                 fallback: str = "google/timesfm-1.0-200m"):
        self.model_id = model_id
        self.num_samples = num_samples
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.fallback = fallback

    def _load(self):
        try:
            from chronos import ChronosPipeline  # type: ignore
            self.pipe_ = ChronosPipeline.from_pretrained(self.model_id, device_map=self.device)
            self.kind_ = "chronos"
            return
        except Exception:
            pass
        try:
            import timesfm  # type: ignore
            self.pipe_ = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if "cuda" in self.device else "cpu",
                    horizon_len=128, num_layers=20),
                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.fallback),
            )
            self.kind_ = "timesfm"
            return
        except Exception as e:
            raise RuntimeError(
                "Install one of: `pip install chronos-forecasting` "
                "or `pip install timesfm` to use ChronosModel. "
                f"Underlying error: {e}")

    def fit(self, y: pd.Series, cov=None):
        self.y_ = y.copy()
        if not hasattr(self, "pipe_"):
            self._load()
        return self

    def predict(self, horizon: int, alpha: float = 0.1, cov=None) -> Forecast:
        import torch
        ctx = torch.tensor(self.y_.values, dtype=torch.float32)
        if self.kind_ == "chronos":
            samples = self.pipe_.predict(
                context=ctx, prediction_length=horizon,
                num_samples=self.num_samples,
            ).cpu().numpy()           # (1, S, H)
            samples = samples[0].T    # → (H, S)
        else:  # timesfm
            mean, q = self.pipe_.forecast(
                inputs=[self.y_.values], horizon_len=horizon)
            # TimesFM returns quantile bands rather than samples — synthesize
            mean = np.asarray(mean[0])
            samples = np.tile(mean[:, None], (1, self.num_samples))
            samples += np.random.default_rng(0).normal(
                0, np.std(self.y_.values - self.y_.shift(1).fillna(method="bfill")),
                samples.shape)
        dist = Empirical(samples=samples)
        lo, hi = dist.interval(alpha)
        return Forecast(mean=dist.mean(), lo=lo, hi=hi, samples=samples, dist=dist)


def _has_cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False