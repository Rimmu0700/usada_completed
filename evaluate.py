import os
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from config import DEVICE, IMAGE_SIZE, CLASS_NAMES, MODEL_LIST, MODEL_INFO, COMPARISON_DIR, get_output_dirs, get_hyperparams
from dataset import get_dataloaders
from model import build_model
from profiler import GPUMemoryTracker, measure_inference_time, measure_flops, get_gpu_info, save_gpu_metrics

# Load the model weights from the best performing checkpoint
def load_best_model(model_name: str, dirs: dict) -> nn.Module:
    model = build_model(model_name)
    best_path = dirs["best_model_path"]
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_path}\nPlease run train.py first for '{model_name}'.")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    # Handle legacy state dict formats
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"[EVAL] Model '{model_name}' loaded successfully from: {best_path}")
    return model

# Run inference over the designated test dataset
def run_test_inference(model: nn.Module, test_loader) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
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
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "peak_vram_mb": mem_tracker.get_peak_mb(),
    }

# Compute statistical performance metrics and generating summary reports
def compute_metrics(results: dict) -> tuple:
    preds = results["preds"]
    labels = results["labels"]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)
    metrics = {
        "accuracy": round(float(acc), 6),
        "f1_macro": round(float(f1), 6),
        "f1_per_class": {cls: round(float(f1_per_class[i]), 6) for i, cls in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist(),
    }
    print("\n======================================================================")
    print("TEST SET EVALUATION RESULTS")
    print("======================================================================")
    print(f"  Test Accuracy      : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test F1-Score      : {f1:.4f}")
    return metrics, report

# Generate visualization plots for confusion matrix analysis
def plot_confusion_matrix(cm: list, save_path: str, display_name: str):
    cm_array = np.array(cm)
    cm_norm = cm_array.astype(float) / (cm_array.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix Count")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.suptitle(f"{display_name} Medical Plant Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# Evaluate a single model architecture and save outputs
def evaluate_single_model(model_name: str) -> dict:
    dirs = get_output_dirs(model_name)
    hp = get_hyperparams(model_name)
    display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
    Path(dirs["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(dirs["plots_dir"]).mkdir(parents=True, exist_ok=True)
    
    print("\n======================================================================")
    print(f"EVALUATING MODEL PIPELINE: {display_name}")
    print("======================================================================")
    
    # Load training history for comparative metrics
    train_res_path = os.path.join(dirs["metrics_dir"], "train_result.json")
    train_hist = {}
    val_stats = {"best_acc": 0, "avg_acc": 0, "low_acc": 0, "best_f1": 0, "avg_f1": 0, "low_f1": 0}
    if os.path.exists(train_res_path):
        with open(train_res_path, "r") as f:
            tr_data = json.load(f)
            train_hist = tr_data.get("history", {})
            val_stats["best_acc"] = tr_data.get("best_val_acc", 0)
            val_stats["avg_acc"] = tr_data.get("avg_val_acc", 0)
            val_stats["low_acc"] = tr_data.get("lowest_val_acc", 0)
            val_stats["best_f1"] = tr_data.get("best_val_f1", 0)
            val_stats["avg_f1"] = tr_data.get("avg_val_f1", 0)
            val_stats["low_f1"] = tr_data.get("lowest_val_f1", 0)
            
    model = load_best_model(model_name, dirs)
    _, _, test_loader = get_dataloaders(batch_size=hp["batch_size"])
    results = run_test_inference(model, test_loader)
    metrics, report = compute_metrics(results)
    inf_time = measure_inference_time(model, n_warmup=10, n_runs=100)
    flops_result = measure_flops(model)
    gpu_info = get_gpu_info()
    
    plot_confusion_matrix(metrics["confusion_matrix"], os.path.join(dirs["plots_dir"], "confusion_matrix.png"), display_name)
    
    # Save text-based evaluation reports
    report_path = os.path.join(dirs["metrics_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{display_name} Medical Plant Detection\n")
        f.write("==================================================\n")
        f.write(f"Test Accuracy : {metrics['accuracy']:.6f}\n")
        f.write(f"Test F1 Macro : {metrics['f1_macro']:.6f}\n\n")
        f.write(report)
        
    all_metrics = {
        "model": model_name,
        "display_name": display_name,
        "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "num_classes": len(CLASS_NAMES),
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "val_stats": val_stats,
        "train_hist": train_hist,
        "inference_time_ms": inf_time,
        "gpu_vram_peak_mb": results["peak_vram_mb"],
        "flops": flops_result,
        "gpu_info": gpu_info,
    }
    
    save_gpu_metrics(all_metrics, dirs["metrics_dir"], "gpu_metrics.json")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_metrics

# Sequentially evaluate all model architectures and generate comparison charts
def evaluate_all_models() -> dict:
    all_results = {}
    for i, model_name in enumerate(MODEL_LIST, start=1):
        all_results[model_name] = evaluate_single_model(model_name)
    _save_comparison_table(all_results)
    return all_results

# Export comprehensive PDF performance reporting
def _save_comparison_table(all_results: dict):
    Path(COMPARISON_DIR).mkdir(parents=True, exist_ok=True)
    pdf_path = os.path.join(COMPARISON_DIR, "comprehensive_evaluation_report.pdf")
    
    try:
        with PdfPages(pdf_path) as pdf:
            # Create summary table visualization
            fig_table, ax_table = plt.subplots(figsize=(18, 5))
            ax_table.axis("off")
            columns = ["Model", "Test Acc", "Val Acc", "Val F1", "Inf Time ms", "VRAM MB", "FLOPs", "Params M"]
            cell_text = []
            
            for model_name, res in all_results.items():
                vs = res["val_stats"]
                acc_str = f"{vs['best_acc']*100:.1f} / {vs['avg_acc']*100:.1f} / {vs['low_acc']*100:.1f}"
                f1_str = f"{vs['best_f1']*100:.1f} / {vs['avg_f1']*100:.1f} / {vs['low_f1']*100:.1f}"
                cell_text.append([
                    res["display_name"],
                    f"{res['test_accuracy']*100:.2f}",
                    acc_str,
                    f1_str,
                    f"{res['inference_time_ms']['mean_ms']:.2f}",
                    f"{res['gpu_vram_peak_mb']:.0f}",
                    res["flops"]["flops_str"],
                    f"{res['flops']['params_M']:.2f}"
                ])
                
            table = ax_table.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Format table aesthetics
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold", color="white")
                    cell.set_facecolor("darkgreen")
                else:
                    if row % 2 == 0:
                        cell.set_facecolor("lightgreen")
                        
            plt.title("Overall Performance Metrics Table", fontsize=16, pad=20, fontweight="bold")
            pdf.savefig(fig_table, bbox_inches="tight")
            plt.close(fig_table)
            
            # Plot comparative bar charts
            model_names = [r["display_name"] for r in all_results.values()]
            test_accuracies = [r["test_accuracy"] * 100 for r in all_results.values()]
            test_f1s = [r["test_f1_macro"] for r in all_results.values()]
            inf_times = [r["inference_time_ms"]["mean_ms"] for r in all_results.values()]
            vram_usages = [r["gpu_vram_peak_mb"] for r in all_results.values()]
            
            fig_chart, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes[0, 0].bar(model_names, test_accuracies, color=["steelblue", "mediumseagreen", "indianred"])
            axes[0, 0].set_title("Test Accuracy Percentage")
            axes[0, 0].grid(axis="y", alpha=0.3)
            axes[0, 1].bar(model_names, test_f1s, color=["steelblue", "mediumseagreen", "indianred"])
            axes[0, 1].set_title("Test F1-Score Macro")
            axes[0, 1].grid(axis="y", alpha=0.3)
            axes[1, 0].bar(model_names, inf_times, color=["steelblue", "mediumseagreen", "indianred"])
            axes[1, 0].set_title("Inference Time milliseconds per image")
            axes[1, 0].grid(axis="y", alpha=0.3)
            axes[1, 1].bar(model_names, vram_usages, color=["steelblue", "mediumseagreen", "indianred"])
            axes[1, 1].set_title("GPU VRAM Peak Usage Megabytes")
            axes[1, 1].grid(axis="y", alpha=0.3)
            plt.suptitle("Architecture Comparison ResNet50 vs MobileNetV3 vs ViT", fontsize=16, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig_chart, bbox_inches="tight")
            plt.close(fig_chart)
            
            # Save individual learning curve plots
            for m_name, res in all_results.items():
                hist = res.get("train_hist", {})
                if not hist or "train_acc" not in hist:
                    continue
                
                epochs = range(1, len(hist["train_acc"]) + 1)
                t_acc = hist["train_acc"]
                v_acc = hist["val_acc"]
                t_err = [1.0 - x for x in t_acc]
                v_err = [1.0 - x for x in v_acc]
                
                fig_curve, axes_curve = plt.subplots(1, 2, figsize=(14, 6))
                
                axes_curve[0].plot(epochs, v_acc, label="Validation Accuracy", color="steelblue", marker="o", markersize=4, linewidth=1.5)
                axes_curve[0].plot(epochs, t_acc, label="Training Accuracy", color="indianred", marker="x", markersize=4, linewidth=1.5)
                axes_curve[0].set_title("Accuracy", fontsize=14)
                axes_curve[0].set_xlabel("Epoch", fontsize=12)
                axes_curve[0].set_ylabel("Accuracy", fontsize=12)
                axes_curve[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
                axes_curve[0].grid(True, linestyle="--", alpha=0.7)
                
                axes_curve[1].plot(epochs, v_err, label="Validation Error", color="steelblue", marker="o", markersize=4, linewidth=1.5)
                axes_curve[1].plot(epochs, t_err, label="Training Error", color="indianred", marker="x", markersize=4, linewidth=1.5)
                axes_curve[1].set_title("Error Rate", fontsize=14)
                axes_curve[1].set_xlabel("Epoch", fontsize=12)
                axes_curve[1].set_ylabel("Error", fontsize=12)
                axes_curve[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
                axes_curve[1].grid(True, linestyle="--", alpha=0.7)
                
                plt.suptitle(f"Learning Curves Epoch count {len(epochs)} for {res['display_name']}", fontsize=15, fontweight="bold")
                plt.subplots_adjust(bottom=0.25)
                pdf.savefig(fig_curve, bbox_inches='tight')
                plt.close(fig_curve)
                
        print(f"\n[SUCCESS] Comprehensive PDF report generated successfully at {pdf_path}")
    except Exception as e:
        print(f"\n[WARNING] Failed to generate PDF report: {e}")

if __name__ == "__main__":
    evaluate_all_models()