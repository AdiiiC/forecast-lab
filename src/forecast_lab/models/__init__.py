"""Model registry and recursive builder.

YAML can address wrappers as first-class:
  - {name: conformal, base: {name: lightgbm, ...}}
  - {name: aci,       base: {name: arima,    ...}, gamma: 0.01}
  - {name: enbpi,     base: {name: prophet,  ...}, B: 20, block_size: 24}
"""
from .naive import SeasonalNaive
from .arima import ARIMAModel
from .prophet_model import ProphetModel
from .lgbm import LightGBMModel
from .nbeats import NBeatsModel
from .deepar import DeepARModel
from .tft import TFTModel
from .patchtst import PatchTSTModel
from .tide import TiDEModel
from .chronos import ChronosModel
from .croston import CrostonModel, TSBModel, ADIDAModel
from .ensemble import EnsembleModel
from .stacking import StackingModel
from .best_of_n import BestOfN
from ..conformal import ConformalWrapper
from ..conformal_adaptive import ACIWrapper, EnbPIWrapper


REGISTRY = {
    "seasonal_naive": SeasonalNaive,
    "arima":          ARIMAModel,
    "prophet":        ProphetModel,
    "lightgbm":       LightGBMModel,
    "nbeats":         NBeatsModel,
    "deepar":         DeepARModel,
    "tft":            TFTModel,
    "patchtst":       PatchTSTModel,
    "tide":           TiDEModel,
    "chronos":        ChronosModel,
    "croston":        CrostonModel,
    "tsb":            TSBModel,
    "adida":          ADIDAModel,
    "ensemble":       EnsembleModel,
    "stacking":       StackingModel,
    "best_of_n":      BestOfN,
}


_WRAPPERS = {"aci", "enbpi", "conformal", "ensemble", "stacking", "best_of_n"}


def build(spec: dict, season_length: int):
    """Recursively build a (possibly wrapped) model from a YAML spec.

    Wrapper specs look like:
        {name: aci|enbpi|conformal, base: {...inner spec...}, ...params}
    """
    spec = dict(spec)
    name = spec.pop("name")

    if name in _WRAPPERS:
        if name in ("ensemble", "stacking", "best_of_n"):
            # Pool wrappers expect a list of sub-specs under "members"
            members = [build(s, season_length=season_length) for s in spec.pop("members", [])]
            if name == "ensemble":
                return EnsembleModel(members, **spec)
            if name == "stacking":
                return StackingModel(members, **spec)
            return BestOfN(members, **spec)

        inner = build(spec.pop("base"), season_length=season_length)
        if name == "aci":
            return ACIWrapper(inner, **spec)
        if name == "enbpi":
            return EnbPIWrapper(inner, **spec)
        return ConformalWrapper(
            inner,
            calibration_size=spec.get("calibration_size", 336),
            horizon=spec.get("horizon", 24),
        )

    cls = REGISTRY[name]
    if name == "seasonal_naive":
        return cls(season_length=season_length)
    return cls(**spec)