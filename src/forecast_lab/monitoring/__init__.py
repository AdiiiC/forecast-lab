"""Production monitoring: residual + coverage drift detection."""
from .monitor import MonitorReport, evaluate

__all__ = ["MonitorReport", "evaluate"]