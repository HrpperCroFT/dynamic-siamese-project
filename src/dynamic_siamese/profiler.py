"""Wall-time + CUDA peak-memory profiler."""
import time
import torch

_STATS = {}

def reset_stats():
    global _STATS
    _STATS = {}

def get_stats():
    return dict(_STATS)

def record(key, value):
    _STATS[key] = value

class Timer:
    """Works as context manager OR via .__enter__()/.__exit__() manually."""

    def __init__(self, device=None):
        if device is not None and not hasattr(device, "type"):
            try:
                device = torch.device(device)
            except Exception:
                device = None
        self.device = device
        self.time_s = 0.0
        self.peak_mb = 0.0
        self._active = False

    def __enter__(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.t0 = time.perf_counter()
        self._active = True
        return self

    def __exit__(self, *args):
        if not self._active:
            return
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        self.time_s = time.perf_counter() - self.t0
        self._active = False

def cuda_empty(device):
    if device is None:
        return
    if not hasattr(device, "type"):
        try:
            device = torch.device(device)
        except Exception:
            return
    if device.type == "cuda":
        torch.cuda.empty_cache()
