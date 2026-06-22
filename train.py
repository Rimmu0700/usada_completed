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
from profiler import GPUMemoryTracker, EpochTimer, get_gpu_info

def setup_environment(dirs: dict):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = False
    for d in [dirs["log_dir"], dirs["checkpoint_dir"], dirs["metrics_dir"], dirs["plots_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\nDevice: {DEVICE}")

def build_class_weights() -> torch.Tensor:
    counts = torch.tensor(CLASS_RAW_COUNTS, dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer, scaler) -> dict:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": round(epoch_loss, 6), "accuracy": round(epoch_acc, 6), "f1": round(epoch_f1, 6)}

def validate_one_epoch(model, loader, criterion) -> dict:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": round(epoch_loss, 6), "accuracy": round(epoch_acc, 6), "f1": round(epoch_f1, 6)}

def maybe_unfreeze(model, model_name: str, epoch: int, base_lr: float):
    schedule = get_unfreeze_schedule(model_name)
    unfreeze_points = {11: 0, 21: 1, 31: 2}
    if epoch not in unfreeze_points:
        return None
    idx = unfreeze_points[epoch]
    if idx >= len(schedule):
        return None
    layer_to_open = schedule[idx]
    unfreeze_layer(model, layer_to_open)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr_factor = 0.1 ** (idx + 1)
    new_optimizer = optim.Adam([
        {"params": trainable_params, "lr": base_lr * lr_factor}
    ], weight_decay=WEIGHT_DECAY)
    return new_optimizer

def train_single_model(model_name: str) -> dict:
    dirs = get_output_dirs(model_name)
    setup_environment(dirs)
    hp = get_hyperparams(model_name)
    batch_size = hp["batch_size"]
    learning_rate = hp["learning_rate"]
    display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
    
    result_path = os.path.join(dirs["metrics_dir"], "train_result.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            res = json.load(f)
        hist = res.get("history", {})
        if "val_acc" in hist and len(hist["val_acc"]) > 0:
            res["best_val_acc"] = max(hist["val_acc"])
            res["lowest_val_acc"] = min(hist["val_acc"])
            res["avg_val_acc"] = round(float(np.mean(hist["val_acc"])), 4)
            res["best_val_f1"] = max(hist["val_f1"])
            res["lowest_val_f1"] = min(hist["val_f1"])
            res["avg_val_f1"] = round(float(np.mean(hist["val_f1"])), 4)
            with open(result_path, "w") as f:
                json.dump(res, f, indent=4)
            print(f"\nModel {display_name} sudah dilatih sebelumnya. Melewati proses training.")
            return res

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    model = build_model(model_name)
    get_model_summary(model)

    if USE_CLASS_WEIGHTED_LOSS:
        class_weights = build_class_weights()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    mem_tracker = GPUMemoryTracker()
    epoch_timer = EpochTimer()

    best_val_loss = float("inf")
    early_stop_count = 0
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "epoch_time_s": [], "gpu_vram_mb": [], "lr": [],
    }

    start_epoch = 1
    checkpoint_path = dirs["final_model_path"]
    if not os.path.exists(checkpoint_path):
        checkpoint_path = dirs["best_model_path"]

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception:
                pass
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
        if "history" in checkpoint:
            history = checkpoint["history"]
        print(f"\nMelanjutkan training dari epoch {start_epoch}")

    total_start = time.perf_counter()
    epoch = start_epoch - 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        new_optimizer = maybe_unfreeze(model, model_name, epoch, learning_rate)
        if new_optimizer is not None:
            optimizer = new_optimizer
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
            )
        mem_tracker.reset()
        epoch_timer.start()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics = validate_one_epoch(model, val_loader, criterion)
        epoch_secs = epoch_timer.stop()
        peak_vram = mem_tracker.get_peak_mb()
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
            f"{display_name} Epoch [{epoch:03d}/{NUM_EPOCHS}] "
            f"Train Acc: {train_metrics['accuracy']:.4f} Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val F1: {val_metrics['f1']:.4f} Time: {epoch_secs:.1f}s VRAM: {peak_vram:.0f}MB"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            early_stop_count = 0
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "history": history,
            }, dirs["best_model_path"])
        else:
            early_stop_count += 1

        if early_stop_count >= EARLY_STOP_PATIENCE:
            break

    if epoch >= start_epoch:
        torch.save({
            "epoch": epoch,
            "model_name": model_name,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        }, dirs["final_model_path"])

    total_time = time.perf_counter() - total_start
    best_val_acc = max(history['val_acc'])
    lowest_val_acc = min(history['val_acc'])
    avg_val_acc = float(np.mean(history['val_acc']))
    best_val_f1 = max(history['val_f1'])
    lowest_val_f1 = min(history['val_f1'])
    avg_val_f1 = float(np.mean(history['val_f1']))
    avg_vram = float(np.mean(history['gpu_vram_mb']))
    peak_vram = float(max(history['gpu_vram_mb']))

    result_json = {
        "model_name": model_name,
        "display_name": display_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "total_epochs_run": epoch,
        "total_time_s": round(total_time, 2),
        "avg_epoch_time_s": epoch_timer.average() if epoch >= start_epoch else 0,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "lowest_val_acc": lowest_val_acc,
        "avg_val_acc": round(avg_val_acc, 4),
        "best_val_f1": best_val_f1,
        "lowest_val_f1": lowest_val_f1,
        "avg_val_f1": round(avg_val_f1, 4),
        "avg_vram_mb": round(avg_vram, 2),
        "peak_vram_mb": peak_vram,
        "history": history,
        "gpu_info": get_gpu_info() if epoch >= start_epoch else {},
    }
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=4)

    _save_plots(history, dirs["plots_dir"], display_name)

    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_json

def train_all_models():
    all_results = {}
    print(f"\nMEMULAI PERBANDINGAN {len(MODEL_LIST)} MODEL")
    for i, model_name in enumerate(MODEL_LIST, start=1):
        result = train_single_model(model_name)
        all_results[model_name] = result

    header = f"{'Model':20s}{'Best Acc':>10s}{'Avg Acc':>10s}{'Low Acc':>10s}{'Best F1':>10s}{'Avg F1':>10s}{'Low F1':>10s}{'Peak VRAM':>12s}"
    print(f"\n{header}")
    print("-" * len(header))
    for model_name, res in all_results.items():
        display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
        print(
            f"{display_name:20s}"
            f"{res.get('best_val_acc', 0):>10.4f}"
            f"{res.get('avg_val_acc', 0):>10.4f}"
            f"{res.get('lowest_val_acc', 0):>10.4f}"
            f"{res.get('best_val_f1', 0):>10.4f}"
            f"{res.get('avg_val_f1', 0):>10.4f}"
            f"{res.get('lowest_val_f1', 0):>10.4f}"
            f"{res.get('peak_vram_mb', 0):>12.1f}"
        )

    from config import COMPARISON_DIR
    Path(COMPARISON_DIR).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(COMPARISON_DIR, "training_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    return all_results

def _save_plots(history: dict, plots_dir: str, display_name: str):
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_acc"], label="Train Acc")
        plt.plot(epochs, history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuracy_curve.png"), dpi=150)
        plt.close()
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=150)
        plt.close()
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["gpu_vram_mb"], color="orange", label="Peak VRAM (MB)")
        plt.xlabel("Epoch")
        plt.ylabel("VRAM (MB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "gpu_usage.png"), dpi=150)
        plt.close()
    except ImportError:
        pass

if __name__ == "__main__":
    train_all_models()