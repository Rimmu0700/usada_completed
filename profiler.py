import time
import json
import os
import torch
import torch.nn as nn
from typing import Optional
from config import DEVICE, IMAGE_SIZE
from pathlib import Path

# Track GPU VRAM usage during model training or inference sessions
class GPUMemoryTracker:
    def __init__(self):
        self.snapshots = []

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def snapshot(self, label: str = ""):
        if not torch.cuda.is_available():
            return {"label": label, "allocated_mb": 0, "reserved_mb": 0}
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        snap = {
            "label": label,
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "peak_mb": round(peak, 2),
        }
        self.snapshots.append(snap)
        return snap

    def get_peak_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / 1024**2, 2)

    def print_current(self, label: str = ""):
        snap = self.snapshot(label)
        print(f"[GPU MEM] {label:30s} | Allocated: {snap['allocated_mb']:7.1f} MB | Peak: {snap['peak_mb']:7.1f} MB")

# Benchmark inference speed in milliseconds per image
def measure_inference_time(model: nn.Module, n_warmup: int = 10, n_runs: int = 100) -> dict:
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
    print(f"[PROFILER] Executing inference warmup ({n_warmup} iterations)...", end=" ")
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                _ = model(dummy)
            except Exception:
                pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    print("Completed.")
    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                timings.append(elapsed_ms)
            except Exception:
                pass
    if not timings:
        print("[WARNING] CUDA Synchronize failed. Using fallback mechanism.")
        timings = [0.0]
    import statistics
    result = {
        "mean_ms": round(statistics.mean(timings), 3),
        "std_ms": round(statistics.stdev(timings), 3) if len(timings) > 1 else 0.0,
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
        "n_runs": len(timings),
    }
    print(f"[PROFILER] Inference Latency -> Mean: {result['mean_ms']:.3f} ms | Std: {result['std_ms']:.3f} ms | Min: {result['min_ms']:.3f} ms | Max: {result['max_ms']:.3f} ms")
    return result

# Calculate model complexity in terms of FLOPs
def measure_flops(model: nn.Module) -> dict:
    try:
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        model.eval()
        with torch.no_grad():
            flops_raw, params_raw = profile(model, inputs=(dummy,), verbose=False)
        flops_str, params_str = clever_format([flops_raw, params_raw], "%.3f")
        result = {
            "flops_raw": flops_raw,
            "flops_str": flops_str,
            "params_raw": params_raw,
            "params_str": params_str,
            "params_M": round(params_raw / 1e6, 2),
            "method": "thop"
        }
    except ImportError:
        total_params = sum(p.numel() for p in model.parameters())
        result = {
            "flops_raw": None,
            "flops_str": "N/A (Please execute: pip install thop)",
            "params_raw": total_params,
            "params_str": f"{total_params/1e6:.2f}M",
            "params_M": round(total_params / 1e6, 2),
            "method": "manual_param_count"
        }
    except Exception as e:
        print(f"[WARNING] Module 'thop' failed to compute FLOPs ({e}). Defaulting to standard parameter count mapping.")
        total_params = sum(p.numel() for p in model.parameters())
        result = {
            "flops_raw": None,
            "flops_str": "N/A (Architecture unsupported by thop module)",
            "params_raw": total_params,
            "params_str": f"{total_params/1e6:.2f}M",
            "params_M": round(total_params / 1e6, 2),
            "method": "manual_param_count_fallback"
        }
    print(f"[PROFILER] Computational FLOPs : {result['flops_str']}")
    print(f"[PROFILER] Total Parameters    : {result['params_str']} ({result['params_M']}M)")
    return result

# Class for tracking and averaging epoch durations
class EpochTimer:
    def __init__(self):
        self.epoch_times = []
        self._start = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.epoch_times.append(elapsed)
        return elapsed

    def average(self) -> float:
        if not self.epoch_times:
            return 0.0
        return round(sum(self.epoch_times) / len(self.epoch_times), 3)

    def total(self) -> float:
        return round(sum(self.epoch_times), 3)

# Query GPU hardware specifications and configuration
def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "device": "CPU"}
    props = torch.cuda.get_device_properties(0)
    info = {
        "available": True,
        "device_name": props.name,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "total_vram_gb": round(props.total_memory / 1024**3, 2),
        "sm_count": props.multi_processor_count,
        "cuda_cores_est": props.multi_processor_count * 128,
        "compute_cap": f"{props.major}.{props.minor}",
        "pytorch_version": torch.__version__,
    }
    print("\n[GPU HARDWARE ARCHITECTURE INFO]")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    return info

# Persist GPU performance metrics to JSON
def save_gpu_metrics(metrics: dict, metrics_dir: str, filename: str = "gpu_metrics.json"):
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, default=str)
    print(f"[PROFILER] GPU hardware allocation metrics saved to: {path}")

if __name__ == "__main__":
    from model import build_model
    from config import get_output_dirs
    model_name = "mobilenet"
    model = build_model(model_name)
    gpu_info = get_gpu_info()
    mem_tracker = GPUMemoryTracker()
    mem_tracker.reset()
    dummy = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy)
    mem_tracker.print_current("Post forward-pass validation (Batch Size = 8)")
    inf_result = measure_inference_time(model, n_warmup=5, n_runs=50)
    flops_result = measure_flops(model)
    all_metrics = {
        "gpu_info": gpu_info,
        "inference_time": inf_result,
        "flops": flops_result,
        "peak_vram_mb": mem_tracker.get_peak_mb(),
    }
    dirs = get_output_dirs(model_name)
    save_gpu_metrics(all_metrics, dirs["metrics_dir"])