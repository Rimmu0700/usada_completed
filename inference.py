"""
inference.py
============
Melakukan prediksi pada SATU gambar tanaman menggunakan salah satu dari
3 model terlatih: ResNet50, MobileNetV3-Small, atau ViT-Base/16.

Alur:
  Path gambar (JPG)
  → Baca dengan PIL
  → Resize 224×224
  → ToTensor + Normalize
  → Tambah dimensi batch: (1, 3, 224, 224)
  → .to(GPU)
  → Forward pass (model yang dipilih)
  → Softmax → probabilitas per kelas
  → Tampilkan prediksi + confidence

Penggunaan:
    python inference.py path/ke/gambar.jpg                  → default: resnet50
    python inference.py path/ke/gambar.jpg --model mobilenet
    python inference.py path/ke/gambar.jpg --model vit
    python inference.py path/ke/gambar.jpg --model all       → bandingkan ketiganya
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from config import DEVICE, CLASS_NAMES, IDX_TO_CLASS, IMAGE_SIZE, MEAN, STD, MODEL_LIST, MODEL_INFO, get_output_dirs
from model import build_model


inference_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def load_model_for_inference(model_name: str) -> nn.Module:
    """Memuat model terbaik (sesuai model_name) untuk inferensi."""
    dirs = get_output_dirs(model_name)
    model = build_model(model_name)

    if not os.path.exists(dirs["best_model_path"]):
        raise FileNotFoundError(
            f"Model '{model_name}' tidak ditemukan: {dirs['best_model_path']}\n"
            f"Jalankan: python train.py  (pastikan '{model_name}' ada di MODEL_LIST config.py)"
        )

    checkpoint = torch.load(dirs["best_model_path"], map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
    print(f"[INFERENCE] Model '{display_name}' dimuat. Val acc: {checkpoint.get('val_acc', float('nan')):.4f}")
    return model


def predict_single_image(model: nn.Module, image_path: str) -> dict:
    """
    Prediksi spesies tanaman dari satu file gambar menggunakan model yang diberikan.

    Return:
        dict berisi predicted_class, confidence, all_probs, inference_ms
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        t_start = time.perf_counter()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(tensor)

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        inference_ms = start_event.elapsed_time(end_event)
    else:
        inference_ms = (time.perf_counter() - t_start) * 1000

    probs      = torch.softmax(logits, dim=1).squeeze(0)
    confidence = probs.max().item()
    pred_idx   = probs.argmax().item()
    pred_class = IDX_TO_CLASS[pred_idx]

    all_probs = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    all_probs_sorted = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return {
        "image_path":    image_path,
        "original_size": original_size,
        "input_size":    f"{IMAGE_SIZE}×{IMAGE_SIZE}",
        "predicted":     pred_class,
        "confidence":    round(confidence * 100, 2),
        "inference_ms":  round(inference_ms, 3),
        "all_probs_%":   all_probs_sorted,
    }


def print_prediction(result: dict, display_name: str = ""):
    """Tampilkan hasil prediksi secara terformat."""
    print("\n" + "=" * 55)
    title = f"HASIL PREDIKSI — {display_name}" if display_name else "HASIL PREDIKSI TANAMAN OBAT"
    print(title)
    print("=" * 55)
    print(f"  Gambar        : {os.path.basename(result['image_path'])}")
    print(f"  Ukuran asli   : {result['original_size']}")
    print(f"  Diproses sbg  : {result['input_size']} px")
    print(f"  Prediksi      : {result['predicted']}")
    print(f"  Confidence    : {result['confidence']:.2f}%")
    print(f"  Inference time: {result['inference_ms']:.3f} ms")
    print("\nProbabilitas semua kelas:")
    for cls, prob in result["all_probs_%"].items():
        bar = "█" * int(prob / 5)
        mark = " ← PREDIKSI" if cls == result["predicted"] else ""
        print(f"  {cls:28s}: {prob:6.2f}% {bar}{mark}")
    print("=" * 55)


def predict_with_all_models(image_path: str):
    """Jalankan prediksi dengan SEMUA model di MODEL_LIST, lalu tampilkan perbandingan."""
    results = {}

    for model_name in MODEL_LIST:
        display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
        try:
            model  = load_model_for_inference(model_name)
            result = predict_single_image(model, image_path)
            print_prediction(result, display_name)
            results[model_name] = result

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"\n[SKIP] {display_name}: {e}")

    if len(results) > 1:
        print("\n\n" + "=" * 70)
        print("PERBANDINGAN PREDIKSI ANTAR MODEL")
        print("=" * 70)
        header = f"{'Model':22s}{'Prediksi':25s}{'Confidence':>12s}{'Inf.Time(ms)':>14s}"
        print(header)
        print("-" * len(header))
        for model_name, res in results.items():
            display_name = MODEL_INFO.get(model_name, {}).get("display_name", model_name)
            print(f"{display_name:22s}{res['predicted']:25s}{res['confidence']:>11.2f}%{res['inference_ms']:>14.3f}")

        predictions = [r["predicted"] for r in results.values()]
        if len(set(predictions)) == 1:
            print(f"\n[KONSENSUS] Semua model sepakat: {predictions[0]}")
        else:
            print(f"\n[BERBEDA] Model tidak sepakat — perlu verifikasi manual.")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi spesies tanaman obat dari satu gambar.")
    parser.add_argument("image_path", help="Path ke file gambar (.jpg/.png)")
    parser.add_argument(
        "--model", default="resnet50", choices=MODEL_LIST + ["all"],
        help="Model yang dipakai: resnet50 (default), mobilenet, vit, atau 'all' untuk bandingkan ketiganya"
    )
    args = parser.parse_args()

    if args.model == "all":
        predict_with_all_models(args.image_path)
    else:
        display_name = MODEL_INFO.get(args.model, {}).get("display_name", args.model)
        model  = load_model_for_inference(args.model)
        result = predict_single_image(model, args.image_path)
        print_prediction(result, display_name)
