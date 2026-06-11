"""GPU and system resource probing."""

from dataclasses import dataclass


@dataclass
class GPUInfo:
    available: bool
    device_count: int
    total_vram_mb: float
    free_vram_mb: float
    device_name: str


@dataclass
class SystemInfo:
    cpu_count: int
    ram_total_mb: float
    ram_available_mb: float


def probe_gpu() -> GPUInfo:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return GPUInfo(False, 0, 0, 0, "")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return GPUInfo(
            available=True,
            device_count=count,
            total_vram_mb=mem.total / (1024 ** 2),
            free_vram_mb=mem.free / (1024 ** 2),
            device_name=name,
        )
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 ** 2)
            free = total - torch.cuda.memory_allocated(0) / (1024 ** 2)
            return GPUInfo(True, torch.cuda.device_count(), total, free, props.name)
    except Exception:
        pass

    return GPUInfo(False, 0, 0, 0, "cpu")


def probe_system() -> SystemInfo:
    import psutil

    mem = psutil.virtual_memory()
    return SystemInfo(
        cpu_count=psutil.cpu_count() or 1,
        ram_total_mb=mem.total / (1024 ** 2),
        ram_available_mb=mem.available / (1024 ** 2),
    )


def snapshot_resources() -> dict:
    gpu = probe_gpu()
    sys = probe_system()
    import psutil

    return {
        "gpu_util_pct": _gpu_util(),
        "gpu_mem_used_mb": gpu.total_vram_mb - gpu.free_vram_mb if gpu.available else 0,
        "cpu_util_pct": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": sys.ram_total_mb - sys.ram_available_mb,
    }


def _gpu_util() -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return 0.0
