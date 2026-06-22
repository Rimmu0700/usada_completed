"""
train.py
========
Training loop untuk MEMBANDINGKAN 3 model secara otomatis berurutan:
  1. ResNet50
  2. MobileNetV3-Small
  3. ViT-Base/16

Setiap model:
  - Dilatih dengan hyperparameter spesifiknya sendiri (lihat MODEL_HYPERPARAMS di config.py)
  - Menggunakan Gradual Unfreeze sesuai jadwal arsitekturnya (lihat UNFREEZE_SCHEDULE di model.py)
  - Hasil (checkpoint, metrics, plots) disimpan ke outputs/<model_name>/ masing-masing
  - GPU memory di-reset total sebelum model berikutnya dimulai (mencegah kebocoran VRAM antar model)

ALUR TRAINING PER EPOCH (sama untuk ketiga model):
  1. Set model ke mode train()
  2. Iterasi batch dari DataLoader
  3. Transfer batch ke GPU (.to(DEVICE))
  4. Forward pass → logits
  5. Hitung loss (CrossEntropyLoss + class weighting otomatis)
  6. Zero gradient → Backward pass → Update weights
  7. Catat loss & accuracy
  8. Validasi di akhir epoch
  9. Simpan checkpoint jika val_loss membaik
"""

import os
import gc
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np

from config import (
    DEVICE, NUM_EPOCHS, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, LR_MIN,
    CUDNN_BENCHMARK, CLASS_NAMES, CLASS_RAW_COUNTS,
    MODEL_LIST, MODEL_INFO, get_output_dirs, get_hyperparams,
    USE_CLASS_WEIGHTED_LOSS,
)
from dataset import get_dataloaders
from model import build_model, get_model_summary, unfreeze_layer, get_unfreeze_schedule
from profiler import GPUMemoryTracker, EpochTimer, get_gpu_info, save_gpu_metrics


# ==============================================================
# SETUP
# ==============================================================

def setup_environment(dirs: dict):
    """Konfigurasi awal CUDA dan buat direktori output untuk model aktif."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = False

    for d in [dirs["log_dir"], dirs["checkpoint_dir"], dirs["metrics_dir"], dirs["plots_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"[SETUP] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[GPU]   {torch.cuda.get_device_name(0)}")
        print(f"[GPU]   VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        print(f"[GPU]   cuDNN benchmark: {CUDNN_BENCHMARK}")


def build_class_weights() -> torch.Tensor:
    """
    Hitung bobot kelas OTOMATIS dari CLASS_RAW_COUNTS di config.py.
    Kelas dengan data lebih sedikit mendapat bobot lebih besar pada loss,
    sehingga model tidak bias ke kelas dengan data lebih banyak.

    weight_i = (1 / count_i) dinormalisasi agar total weight = 1
    """
    counts = torch.tensor(CLASS_RAW_COUNTS, dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights.to(DEVICE)


# ==============================================================
# SATU EPOCH TRAINING
# ==============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler) -> dict:
    """
    Satu epoch training dengan Mixed Precision (AMP).
    """
    model.train()

    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    for images, labels in loader:
        # [GPU] Transfer batch ke VRAM
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # [GPU] Mixed Precision forward pass
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds         = torch.argmax(outputs, dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    epoch_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": round(epoch_loss, 6), "accuracy": round(epoch_acc, 6), "f1": round(epoch_f1, 6)}


# ==============================================================
# SATU EPOCH VALIDASI
# ==============================================================

def validate_one_epoch(model, loader, criterion) -> dict:
    """Evaluasi model pada validation set, tanpa gradient computation."""
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds         = torch.argmax(outputs, dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    epoch_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": round(epoch_loss, 6), "accuracy": round(epoch_acc, 6), "f1": round(epoch_f1, 6)}


# ==============================================================
# GRADUAL UNFREEZE HANDLER
# ==============================================================

def maybe_unfreeze(model, model_name: str, epoch: int, base_lr: float):
    """
    Cek apakah epoch saat ini adalah titik untuk membuka layer baru.
    Jadwal: epoch 11 → buka layer terdalam ke-1, epoch 21 → ke-2, epoch 31 → ke-3.

    Jika ada layer yang dibuka, optimizer DIBUAT ULANG dengan param groups
    baru, supaya optimizer Adam tidak menyimpan momentum dari konfigurasi lama
    yang sudah tidak sesuai jumlah parameter trainable.

    Return:
        optimizer baru jika terjadi unfreeze, None jika tidak ada perubahan.
    """
    schedule = get_unfreeze_schedule(model_name)
    unfreeze_points = {11: 0, 21: 1, 31: 2}

    if epoch not in unfreeze_points:
        return None

    idx = unfreeze_points[epoch]
    if idx >= len(schedule):
        return None

    layer_to_open = schedule[idx]
    unfreeze_layer(model, layer_to_open)

    # Param groups: layer yang baru dibuka mendapat lr lebih kecil
    # dibanding classifier head, supaya bobot pretrained tidak rusak drastis.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr_factor = 0.1 ** (idx + 1)   # semakin dalam unfreeze, lr semakin kecil

    new_optimizer = optim.Adam([
        {"params": trainable_params, "lr": base_lr * lr_factor}
    ], weight_decay=WEIGHT_DECAY)

    print(f"[TRAIN] Optimizer di-reset setelah unfreeze '{layer_to_open}' "
          f"(lr layer baru: {base_lr * lr_factor:.8f})")

    return new_optimizer


# ==============================================================
# TRAINING SATU MODEL
# ==============================================================

def train_single_model(model_name: str) -> dict:
    """
    Jalankan seluruh proses training untuk SATU model.
    Dipanggil berulang dari train_all_models() untuk setiap model di MODEL_LIST.
    """
    dirs = get_output_dirs(model_name)
    setup_environment(dirs)

    hp = get_hyperparams(model_name)
    batch_size    = hp["batch_size"]
    learning_rate = hp["learning_rate"]

    display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)

    print("\n" + "#" * 70)
    print(f"#  TRAINING MODEL: {display_name}")
    print(f"#  Batch size: {batch_size} | Learning rate: {learning_rate}")
    print("#" * 70)

    # --- Data ---
    print("\n" + "=" * 65)
    print("MEMUAT DATASET")
    print("=" * 65)
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # --- Model ---
    print("\n" + "=" * 65)
    print(f"MEMBANGUN MODEL: {display_name}")
    print("=" * 65)
    model = build_model(model_name)
    get_model_summary(model)

    # --- Loss dengan class weighting otomatis ---
    if USE_CLASS_WEIGHTED_LOSS:
        class_weights = build_class_weights()
        print(f"[INFO] Class weights (otomatis dari CLASS_RAW_COUNTS): "
              f"{class_weights.cpu().numpy().round(4)}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Optimizer (hanya parameter trainable di awal: classifier head) ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())

    mem_tracker = GPUMemoryTracker()
    epoch_timer = EpochTimer()

    best_val_loss    = float("inf")
    early_stop_count = 0
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss":   [], "val_acc":   [], "val_f1":   [],
        "epoch_time_s": [], "gpu_vram_mb": [], "lr": [],
    }

    print("\n" + "=" * 65)
    print(f"MULAI TRAINING — {display_name} — {NUM_EPOCHS} epochs")
    print(f"Device : {DEVICE}")
    print("=" * 65)

    total_start = time.perf_counter()
    epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        # [EXPERIMENT] Gradual unfreeze — cek apakah epoch ini saatnya buka layer baru
        new_optimizer = maybe_unfreeze(model, model_name, epoch, learning_rate)
        if new_optimizer is not None:
            optimizer = new_optimizer
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
            )

        mem_tracker.reset()
        epoch_timer.start()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics   = validate_one_epoch(model, val_loader, criterion)

        epoch_secs = epoch_timer.stop()
        peak_vram  = mem_tracker.get_peak_mb()

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["epoch_time_s"].append(epoch_secs)
        history["gpu_vram_mb"].append(peak_vram)
        history["lr"].append(current_lr)

        print(
            f"[{display_name}] Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"VRAM: {peak_vram:.0f}MB | Time: {epoch_secs:.1f}s | LR: {current_lr:.8f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss    = val_metrics["loss"]
            early_stop_count = 0
            torch.save({
                "epoch":       epoch,
                "model_name":  model_name,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "val_acc":     val_metrics["accuracy"],
                "val_f1":      val_metrics["f1"],
                "history":     history,
            }, dirs["best_model_path"])
            print(f"  ✓ Best model disimpan (val_loss: {best_val_loss:.6f})")
        else:
            early_stop_count += 1

        if early_stop_count >= EARLY_STOP_PATIENCE:
            print(f"\n[EARLY STOP] Val loss tidak membaik selama {EARLY_STOP_PATIENCE} epoch. Berhenti.")
            break

    torch.save({
        "epoch":       epoch,
        "model_name":  model_name,
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "history":     history,
    }, dirs["final_model_path"])

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 65)
    print(f"TRAINING SELESAI — {display_name}")
    print(f"  Total waktu          : {total_time/60:.1f} menit")
    print(f"  Avg waktu/epoch      : {epoch_timer.average():.2f} detik")
    print(f"  Best val loss        : {best_val_loss:.6f}")
    print(f"  Best val acc         : {max(history['val_acc']):.4f}")
    print(f"  Best val F1          : {max(history['val_f1']):.4f}")
    print(f"  Avg VRAM usage       : {np.mean(history['gpu_vram_mb']):.1f} MB")
    print(f"  Peak VRAM            : {max(history['gpu_vram_mb']):.1f} MB")
    print("=" * 65)

    result_json = {
        "model_name":         model_name,
        "display_name":       display_name,
        "batch_size":         batch_size,
        "learning_rate":      learning_rate,
        "total_epochs_run":   epoch,
        "total_time_s":       round(total_time, 2),
        "avg_epoch_time_s":   epoch_timer.average(),
        "best_val_loss":      best_val_loss,
        "best_val_acc":       max(history["val_acc"]),
        "best_val_f1":        max(history["val_f1"]),
        "avg_vram_mb":        round(float(np.mean(history["gpu_vram_mb"])), 2),
        "peak_vram_mb":       max(history["gpu_vram_mb"]),
        "history":            history,
        "gpu_info":           get_gpu_info(),
    }
    result_path = os.path.join(dirs["metrics_dir"], "train_result.json")
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=4)
    print(f"[LOG] Training result disimpan → {result_path}")

    _save_plots(history, dirs["plots_dir"], display_name)

    # [GPU] Bersihkan model dari VRAM sebelum model berikutnya dimulai
    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_json


# ==============================================================
# LOOP SEMUA MODEL
# ==============================================================

def train_all_models():
    """
    Jalankan training untuk SEMUA model di MODEL_LIST secara berurutan.
    Setelah semua selesai, tampilkan tabel perbandingan ringkas.
    """
    all_results = {}

    print("\n" + "█" * 70)
    print(f"█  MEMULAI PERBANDINGAN {len(MODEL_LIST)} MODEL: {', '.join(MODEL_LIST)}")
    print("█" * 70)

    for i, model_name in enumerate(MODEL_LIST, start=1):
        print(f"\n\n>>> MODEL {i}/{len(MODEL_LIST)}: {model_name} <<<")
        result = train_single_model(model_name)
        all_results[model_name] = result

    # --- Tabel perbandingan ringkas ---
    print("\n\n" + "█" * 70)
    print("█  RINGKASAN PERBANDINGAN SEMUA MODEL")
    print("█" * 70)
    header = f"{'Model':22s}{'Best Acc':>10s}{'Best F1':>10s}{'Avg s/epoch':>14s}{'Peak VRAM(MB)':>16s}"
    print(header)
    print("-" * len(header))
    for model_name, res in all_results.items():
        display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
        print(
            f"{display_name:22s}"
            f"{res['best_val_acc']:>10.4f}"
            f"{res['best_val_f1']:>10.4f}"
            f"{res['avg_epoch_time_s']:>14.2f}"
            f"{res['peak_vram_mb']:>16.1f}"
        )

    # Simpan ringkasan gabungan
    from config import COMPARISON_DIR
    Path(COMPARISON_DIR).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(COMPARISON_DIR, "training_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[LOG] Perbandingan training lengkap disimpan → {summary_path}")

    return all_results


# ==============================================================
# PLOTTING HELPER
# ==============================================================

def _save_plots(history: dict, plots_dir: str, display_name: str):
    """Simpan grafik accuracy, loss, dan GPU VRAM usage untuk satu model."""
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_acc"], label="Train Acc")
        plt.plot(epochs, history["val_acc"],   label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curve — {display_name}"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuracy_curve.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Loss Curve — {display_name}"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["gpu_vram_mb"], color="orange", label="Peak VRAM (MB)")
        plt.xlabel("Epoch"); plt.ylabel("VRAM (MB)")
        plt.title(f"GPU VRAM Usage — {display_name}"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gpu_usage.png"), dpi=150)
        plt.close()

        print(f"[PLOT] Grafik disimpan → {plots_dir}/")

    except ImportError:
        print("[WARNING] matplotlib tidak tersedia. Plot dilewati.")


if __name__ == "__main__":
    train_all_models()
