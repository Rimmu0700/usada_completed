"""
evaluate.py
===========
Evaluasi SEMUA model (ResNet50, MobileNetV3-Small, ViT-Base/16) pada test set,
dengan metrik penelitian lengkap untuk masing-masing:

  1. Accuracy & F1-Score   → kualitas prediksi
  2. Inference Time        → ms per gambar (diukur di GPU, batch=1 untuk semua model)
  3. GPU Memory Usage      → VRAM peak saat inferensi
  4. FLOPs & Parameters    → kompleksitas komputasi model
  5. Confusion Matrix      → visualisasi per-kelas
  6. Classification Report → precision, recall, F1 per spesies

Hasil per model disimpan ke outputs/<model_name>/metrics/ dan outputs/<model_name>/plots/
Tabel perbandingan akhir disimpan ke outputs/comparison/
"""

import os
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEVICE, IMAGE_SIZE, CLASS_NAMES,
    MODEL_LIST, MODEL_INFO, COMPARISON_DIR,
    get_output_dirs, get_hyperparams,
)
from dataset import get_dataloaders
from model import build_model
from profiler import (
    GPUMemoryTracker,
    measure_inference_time,
    measure_flops,
    get_gpu_info,
    save_gpu_metrics,
)


def load_best_model(model_name: str, dirs: dict) -> nn.Module:
    """Memuat model dengan bobot terbaik (checkpoint best_model.pth) untuk model_name."""
    model = build_model(model_name)
    best_path = dirs["best_model_path"]

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {best_path}\n"
            f"Jalankan train.py terlebih dahulu untuk model '{model_name}'."
        )

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"[EVAL] Model '{model_name}' dimuat dari: {best_path}")
    print(f"[EVAL] Checkpoint epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"[EVAL] Val accuracy     : {checkpoint.get('val_acc', float('nan')):.4f}")
    print(f"[EVAL] Val F1           : {checkpoint.get('val_f1', float('nan')):.4f}")

    return model


def run_test_inference(model: nn.Module, test_loader) -> dict:
    """Jalankan inferensi seluruh test set, kumpulkan prediksi & probabilitas."""
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    mem_tracker = GPUMemoryTracker()
    mem_tracker.reset()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "preds":  np.array(all_preds),
        "labels": np.array(all_labels),
        "probs":  np.array(all_probs),
        "peak_vram_mb": mem_tracker.get_peak_mb(),
    }


def compute_metrics(results: dict) -> tuple:
    """Hitung semua metrik evaluasi dari hasil inferensi test set."""
    preds  = results["preds"]
    labels = results["labels"]

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    cm  = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)

    metrics = {
        "accuracy":    round(float(acc), 6),
        "f1_macro":    round(float(f1), 6),
        "f1_per_class": {
            cls: round(float(f1_per_class[i]), 6)
            for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }

    print("\n" + "=" * 65)
    print("HASIL EVALUASI TEST SET")
    print("=" * 65)
    print(f"  Accuracy (Overall) : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score (Macro)   : {f1:.4f}")
    print("\nF1-Score per Spesies:")
    for cls, f1v in metrics["f1_per_class"].items():
        bar = "█" * int(f1v * 20)
        print(f"  {cls:28s}: {f1v:.4f} {bar}")
    print("\nClassification Report:")
    print(report)

    return metrics, report


def plot_confusion_matrix(cm: list, save_path: str, display_name: str):
    """Visualisasi confusion matrix dan simpan sebagai PNG."""
    cm_array = np.array(cm)
    cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Count)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.suptitle(f"{display_name} — Medical Plant Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrix disimpan → {save_path}")


def evaluate_single_model(model_name: str) -> dict:
    """Evaluasi lengkap untuk SATU model. Dipanggil berulang oleh evaluate_all_models()."""
    dirs = get_output_dirs(model_name)
    hp   = get_hyperparams(model_name)
    display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)

    Path(dirs["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(dirs["plots_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"#  EVALUASI MODEL: {display_name}")
    print("#" * 70)

    model = load_best_model(model_name, dirs)
    # Test loader batch=1 selalu, terlepas dari batch_size training model ini
    _, _, test_loader = get_dataloaders(batch_size=hp["batch_size"])

    print(f"\n[EVAL] Menjalankan inferensi pada test set...")
    results = run_test_inference(model, test_loader)
    metrics, report = compute_metrics(results)

    print("\n[PROFILER] Mengukur inference time...")
    inf_time = measure_inference_time(model, n_warmup=10, n_runs=100)

    print("[PROFILER] Menghitung FLOPs & parameters...")
    flops_result = measure_flops(model)

    gpu_info = get_gpu_info()

    cm_path = os.path.join(dirs["plots_dir"], "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path, display_name)

    report_path = os.path.join(dirs["metrics_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{display_name} — Medical Plant Detection\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy : {metrics['accuracy']:.6f}\n")
        f.write(f"F1 Macro : {metrics['f1_macro']:.6f}\n\n")
        f.write(report)
    print(f"[LOG] Classification report → {report_path}")

    all_metrics = {
        "model":              model_name,
        "display_name":       display_name,
        "image_size":         f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "num_classes":        len(CLASS_NAMES),
        "accuracy":           metrics["accuracy"],
        "f1_macro":           metrics["f1_macro"],
        "f1_per_class":       metrics["f1_per_class"],
        "inference_time_ms":  inf_time,
        "gpu_vram_peak_mb":   results["peak_vram_mb"],
        "flops":              flops_result,
        "gpu_info":           gpu_info,
    }

    save_gpu_metrics(all_metrics, dirs["metrics_dir"], "gpu_metrics.json")

    print("\n" + "=" * 65)
    print(f"RINGKASAN METRIK — {display_name}")
    print("=" * 65)
    print(f"  Input Resolution   : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Accuracy           : {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score (Macro)   : {metrics['f1_macro']:.4f}")
    print(f"  Inference Time     : {inf_time['mean_ms']:.3f} ms ± {inf_time['std_ms']:.3f} ms")
    print(f"  GPU VRAM (peak)    : {results['peak_vram_mb']:.1f} MB")
    print(f"  FLOPs              : {flops_result['flops_str']}")
    print(f"  Parameters         : {flops_result['params_str']}")
    print(f"  GPU                : {gpu_info.get('device_name','N/A')}")
    print("=" * 65)

    # [GPU] Bersihkan VRAM sebelum model berikutnya
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_metrics


def evaluate_all_models() -> dict:
    """Evaluasi SEMUA model di MODEL_LIST, lalu buat tabel perbandingan akhir."""
    all_results = {}

    for i, model_name in enumerate(MODEL_LIST, start=1):
        print(f"\n\n>>> EVALUASI MODEL {i}/{len(MODEL_LIST)}: {model_name} <<<")
        all_results[model_name] = evaluate_single_model(model_name)

    _print_comparison_table(all_results)
    _save_comparison_table(all_results)

    return all_results


def _print_comparison_table(all_results: dict):
    """Cetak tabel perbandingan ringkas semua model — siap untuk paper."""
    print("\n\n" + "█" * 90)
    print("█  TABEL PERBANDINGAN AKHIR — SEMUA MODEL")
    print("█" * 90)

    header = (
        f"{'Model':22s}{'Accuracy':>10s}{'F1-Macro':>10s}"
        f"{'Inf.Time(ms)':>14s}{'VRAM(MB)':>11s}{'FLOPs':>12s}{'Params(M)':>11s}"
    )
    print(header)
    print("-" * len(header))

    for model_name, res in all_results.items():
        print(
            f"{res['display_name']:22s}"
            f"{res['accuracy']*100:>9.2f}%"
            f"{res['f1_macro']:>10.4f}"
            f"{res['inference_time_ms']['mean_ms']:>14.3f}"
            f"{res['gpu_vram_peak_mb']:>11.1f}"
            f"{res['flops']['flops_str']:>12s}"
            f"{res['flops']['params_M']:>11.2f}"
        )
    print("█" * 90)


def _save_comparison_table(all_results: dict):
    """Simpan tabel perbandingan ke JSON dan buat bar chart visual."""
    Path(COMPARISON_DIR).mkdir(parents=True, exist_ok=True)

    json_path = os.path.join(COMPARISON_DIR, "evaluation_comparison.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"\n[LOG] Tabel perbandingan evaluasi disimpan → {json_path}")

    # Bar chart perbandingan visual (4 metrik utama)
    try:
        model_names   = [r["display_name"] for r in all_results.values()]
        accuracies    = [r["accuracy"] * 100 for r in all_results.values()]
        f1_scores     = [r["f1_macro"] for r in all_results.values()]
        inf_times     = [r["inference_time_ms"]["mean_ms"] for r in all_results.values()]
        vram_usages   = [r["gpu_vram_peak_mb"] for r in all_results.values()]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].bar(model_names, accuracies, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[0, 0].set_title("Accuracy (%)"); axes[0, 0].set_ylabel("%")
        axes[0, 0].grid(axis="y", alpha=0.3)

        axes[0, 1].bar(model_names, f1_scores, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[0, 1].set_title("F1-Score (Macro)"); axes[0, 1].set_ylabel("F1")
        axes[0, 1].grid(axis="y", alpha=0.3)

        axes[1, 0].bar(model_names, inf_times, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[1, 0].set_title("Inference Time (ms/gambar)"); axes[1, 0].set_ylabel("ms")
        axes[1, 0].grid(axis="y", alpha=0.3)

        axes[1, 1].bar(model_names, vram_usages, color=["#4C72B0", "#55A868", "#C44E52"])
        axes[1, 1].set_title("GPU VRAM Peak Usage (MB)"); axes[1, 1].set_ylabel("MB")
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.suptitle("Perbandingan Performa: ResNet50 vs MobileNetV3-Small vs ViT-Base/16", fontsize=14)
        plt.tight_layout()
        chart_path = os.path.join(COMPARISON_DIR, "model_comparison_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Bar chart perbandingan disimpan → {chart_path}")

    except Exception as e:
        print(f"[WARNING] Gagal membuat bar chart perbandingan: {e}")


if __name__ == "__main__":
    evaluate_all_models()
