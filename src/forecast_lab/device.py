"""Adaptive compute-device and host-profile helpers.

The YAML configs in this project were originally tuned for a Windows box with an
NVIDIA GPU (see the VRAM / core-count comments in ``configs/energy_cov.yaml``).
This module lets the same configs run unchanged on Apple-Silicon Macs, plain-CPU
hosts, or the original CUDA machine by resolving the device and worker settings
at runtime instead of hard-coding them.

Resolution order for the compute device is CUDA, then Apple MPS, then CPU. Set
the ``FORECAST_LAB_DEVICE`` environment variable to force a specific device.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def pick_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    Order: explicit ``prefer`` / ``FORECAST_LAB_DEVICE`` env, then ``cuda``,
    then Apple ``mps``, then ``cpu``.
    """
    override = prefer or os.environ.get("FORECAST_LAB_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class SystemProfile:
    device: str
    cpu_count: int
    total_ram_gb: float
    gpu_name: str
    gpu_memory_gb: float

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device == "mps"


def _total_ram_gb() -> float:
    """Return total physical RAM in GB, cross-platform (Linux/macOS/Windows)."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / 1e9, 1)
    except Exception:
        pass
    return 0.0


def system_profile() -> SystemProfile:
    """Detect the host's compute device, CPU count, RAM, and GPU details."""
    device = pick_device()
    cpu_count = os.cpu_count() or 4
    gpu_name = ""
    gpu_memory_gb = 0.0
    try:
        import torch

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
        elif device == "mps":
            gpu_name = "Apple MPS"
            # Apple silicon shares unified memory with the CPU.
            gpu_memory_gb = _total_ram_gb()
    except Exception:
        pass
    return SystemProfile(
        device=device,
        cpu_count=cpu_count,
        total_ram_gb=_total_ram_gb(),
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


def adapt_config(cfg: dict) -> SystemProfile:
    """Adapt a loaded config dict in-place to the current host.

    Adjusts parallel worker counts and per-model CPU-thread caps to the host's
    core count, and enables the MPS op fallback so models that use ops not yet
    implemented on Apple GPUs degrade to CPU for those ops instead of crashing.
    Returns the detected :class:`SystemProfile`.
    """
    profile = system_profile()

    if profile.is_mps:
        # Let unsupported ops fall back to CPU instead of raising.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    par = cfg.get("parallel")
    if isinstance(par, dict) and par.get("enabled"):
        requested = par.get("n_workers")
        # Leave one core for the OS / main process; never exceed cores available.
        ceiling = max(1, profile.cpu_count - 1)
        par["n_workers"] = min(requested, ceiling) if requested else ceiling

    thread_cap = max(1, profile.cpu_count)
    for spec in _iter_model_specs(cfg):
        if "cpu_threads" in spec:
            spec["cpu_threads"] = min(int(spec["cpu_threads"]), thread_cap)

    return profile


def _iter_model_specs(cfg: dict):
    """Yield every model spec dict in a config, including nested ``base`` models."""
    specs = list(cfg.get("models", []))
    base = cfg.get("base_model")
    if isinstance(base, dict):
        specs.append(base)
    while specs:
        spec = specs.pop()
        if not isinstance(spec, dict):
            continue
        yield spec
        nested = spec.get("base")
        if isinstance(nested, dict):
            specs.append(nested)