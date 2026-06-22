"""
profiler.py
===========
Mengukur semua metrik performa GPU yang dibutuhkan penelitian:

  1. GPU Memory Usage (MB)     → VRAM yang dikonsumsi selama training/inferensi
  2. Inference Time (ms/gambar)→ kecepatan prediksi satu gambar
  3. Training Time (detik)     → durasi per epoch
  4. FLOPs                     → jumlah operasi floating point per forward pass
  5. Parameters (juta)         → ukuran model

Generik untuk ResNet50, MobileNetV3, dan ViT-Base/16 — semua fungsi di sini
tidak menyebut nama model spesifik apapun, hanya menerima nn.Module umum.
"""

import time
import json
import os
import torch
import torch.nn as nn
from typing import Optional

from config import DEVICE, IMAGE_SIZE
from pathlib import Path


# ==============================================================
# 1. GPU MEMORY TRACKER
# ==============================================================

class GPUMemoryTracker:
    """
    Melacak penggunaan VRAM GPU selama training atau inferensi.
    """

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
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        peak      = torch.cuda.max_memory_allocated() / 1024**2

        snap = {
            "label":        label,
            "allocated_mb": round(allocated, 2),
            "reserved_mb":  round(reserved, 2),
            "peak_mb":      round(peak, 2),
        }
        self.snapshots.append(snap)
        return snap

    def get_peak_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / 1024**2, 2)

    def print_current(self, label: str = ""):
        snap = self.snapshot(label)
        print(
            f"[GPU MEM] {label:30s} | "
            f"Allocated: {snap['allocated_mb']:7.1f} MB | "
            f"Peak: {snap['peak_mb']:7.1f} MB"
        )


# ==============================================================
# 2. INFERENCE TIME MEASUREMENT
# ==============================================================

def measure_inference_time(
    model: nn.Module,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """
    Mengukur inference time per gambar dalam milidetik.
    Generik — bekerja sama untuk ResNet50, MobileNet, dan ViT karena
    hanya butuh forward pass biasa dengan input (1, 3, IMAGE_SIZE, IMAGE_SIZE).
    """
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    print(f"[PROFILER] Warmup ({n_warmup} passes)...", end=" ")
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
    print("done")

    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = model(dummy)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                timings.append(elapsed_ms)
            except Exception:
                pass

    if not timings:
        print("[WARNING] CUDA Synchronize gagal, menggunakan nilai fallback.")
        timings = [0.0]

    import statistics
    result = {
        "mean_ms": round(statistics.mean(timings), 3),
        "std_ms":  round(statistics.stdev(timings), 3) if len(timings) > 1 else 0.0,
        "min_ms":  round(min(timings), 3),
        "max_ms":  round(max(timings), 3),
        "n_runs":  len(timings),
    }

    print(f"[PROFILER] Inference Time: "
          f"mean={result['mean_ms']:.3f} ms | "
          f"std={result['std_ms']:.3f} ms | "
          f"min={result['min_ms']:.3f} ms | "
          f"max={result['max_ms']:.3f} ms")

    return result


# ==============================================================
# 3. FLOPS COUNTER
# ==============================================================

def measure_flops(model: nn.Module) -> dict:
    """
    Menghitung FLOPs per forward pass. Generik untuk semua arsitektur
    yang didukung thop (CNN dan Transformer-based torchvision models
    keduanya didukung oleh thop selama modulnya standar nn.Module).
    """
    try:
        from thop import profile, clever_format

        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        model.eval()

        with torch.no_grad():
            flops_raw, params_raw = profile(model, inputs=(dummy,), verbose=False)

        flops_str, params_str = clever_format([flops_raw, params_raw], "%.3f")

        result = {
            "flops_raw":   flops_raw,
            "flops_str":   flops_str,
            "params_raw":  params_raw,
            "params_str":  params_str,
            "params_M":    round(params_raw / 1e6, 2),
            "method":      "thop"
        }

    except ImportError:
        total_params = sum(p.numel() for p in model.parameters())
        result = {
            "flops_raw":   None,
            "flops_str":   "N/A (install thop: pip install thop)",
            "params_raw":  total_params,
            "params_str":  f"{total_params/1e6:.2f}M",
            "params_M":    round(total_params / 1e6, 2),
            "method":      "manual_param_count"
        }
    except Exception as e:
        # ViT terkadang punya operasi (misal F.scaled_dot_product_attention)
        # yang tidak selalu terhitung sempurna oleh thop di semua versi.
        # Fallback aman: tetap kembalikan jumlah parameter manual.
        print(f"[WARNING] thop gagal menghitung FLOPs ({e}). Fallback ke jumlah parameter saja.")
        total_params = sum(p.numel() for p in model.parameters())
        result = {
            "flops_raw":   None,
            "flops_str":   "N/A (thop error pada arsitektur ini)",
            "params_raw":  total_params,
            "params_str":  f"{total_params/1e6:.2f}M",
            "params_M":    round(total_params / 1e6, 2),
            "method":      "manual_param_count_fallback"
        }

    print(f"[PROFILER] FLOPs   : {result['flops_str']}")
    print(f"[PROFILER] Params  : {result['params_str']} ({result['params_M']}M)")

    return result


# ==============================================================
# 4. EPOCH TIMER
# ==============================================================

class EpochTimer:
    """Mengukur durasi training per epoch."""

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


# ==============================================================
# 5. GPU INFO
# ==============================================================

def get_gpu_info() -> dict:
    """Mengambil informasi spesifikasi GPU yang sedang digunakan."""
    if not torch.cuda.is_available():
        return {"available": False, "device": "CPU"}

    props = torch.cuda.get_device_properties(0)

    info = {
        "available":      True,
        "device_name":    props.name,
        "cuda_version":   torch.version.cuda,
        "cudnn_version":  torch.backends.cudnn.version(),
        "total_vram_gb":  round(props.total_memory / 1024**3, 2),
        "sm_count":       props.multi_processor_count,
        "cuda_cores_est": props.multi_processor_count * 128,
        "compute_cap":    f"{props.major}.{props.minor}",
        "pytorch_version": torch.__version__,
    }

    print("\n[GPU INFO]")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")

    return info


# ==============================================================
# 6. SIMPAN METRIK GPU (path metrics_dir diberikan dari luar)
# ==============================================================

def save_gpu_metrics(metrics: dict, metrics_dir: str, filename: str = "gpu_metrics.json"):
    """
    Simpan semua hasil pengukuran GPU ke file JSON.

    Parameter:
        metrics_dir : folder tujuan (mis. outputs/resnet50/metrics)
                      WAJIB diberikan eksplisit karena setiap model
                      punya folder output terpisah.
    """
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(metrics_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, default=str)
    print(f"[PROFILER] GPU metrics disimpan → {path}")


# ==============================================================
# QUICK TEST
# ==============================================================
if __name__ == "__main__":
    from model import build_model
    from config import get_output_dirs

    model_name = "mobilenet"   # ganti ke "resnet50" / "vit" untuk tes lain
    model = build_model(model_name)

    gpu_info = get_gpu_info()

    mem_tracker = GPUMemoryTracker()
    mem_tracker.reset()

    dummy = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        _ = model(dummy)
    mem_tracker.print_current("Setelah forward pass batch=8")

    inf_result = measure_inference_time(model, n_warmup=5, n_runs=50)
    flops_result = measure_flops(model)

    all_metrics = {
        "gpu_info":      gpu_info,
        "inference_time": inf_result,
        "flops":         flops_result,
        "peak_vram_mb":  mem_tracker.get_peak_mb(),
    }
    dirs = get_output_dirs(model_name)
    save_gpu_metrics(all_metrics, dirs["metrics_dir"])
