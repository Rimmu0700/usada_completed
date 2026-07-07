import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# --- Import YOLO dari ultralytics ---
from ultralytics import YOLO

# Import variabel dari config
from config import DEVICE, CLASS_NAMES, IDX_TO_CLASS, IMAGE_SIZE, MEAN, STD, MODEL_LIST, get_output_dirs, YOLO_WEIGHTS_PATH, YOLO_CONF_THRESHOLD
from model import build_model

# Inisialisasi model YOLO dari config
yolo_model = YOLO(YOLO_WEIGHTS_PATH)

inference_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

def load_model_for_inference(model_name: str) -> nn.Module:
    dirs = get_output_dirs(model_name)
    checkpoint_path = dirs["best_model_path"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint untuk {model_name} tidak ditemukan di: {checkpoint_path}")

    model = build_model(model_name)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

def predict_single_image(model: nn.Module, image_path: str) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    # --- Deteksi Area Daun dengan YOLO ---
    img_np = np.array(img)
    yolo_results = yolo_model(img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = yolo_results[0].boxes

    if len(boxes) > 0:
        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cropped_img_np = img_np[y1:y2, x1:x2]
        img = Image.fromarray(cropped_img_np)
        yolo_status = f"Terdeteksi Daun (Crop: {img.size[0]}x{img.size[1]})"
    else:
        yolo_status = "Daun tidak ditemukan oleh YOLO (Menggunakan Gambar Penuh)"
    # -------------------------------------

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
        "yolo_status":   yolo_status
    }

def print_prediction(result: dict, display_name: str = ""):
    print("\n" + "=" * 60)
    title = f"HASIL PREDIKSI SISTEM — {display_name}" if display_name else "HASIL PREDIKSI DAUN OBAT"
    print(title)
    print("=" * 60)
    print(f"  Nama File     : {os.path.basename(result['image_path'])}")
    print(f"  Ukuran Asli   : {result['original_size'][0]}x{result['original_size'][1]} px")
    print(f"  Sistem YOLO11 : {result['yolo_status']}")
    print(f"  Ukuran Klasif : {result['input_size']}")
    print(f"  Hasil Prediksi: {result['predicted']}")
    print(f"  Confidence    : {result['confidence']:.2f}%")
    print(f"  Waktu Proses  : {result['inference_ms']:.3f} ms")
    print("-" * 60)
    print("Probabilitas Distribusi Kelas:")
    for cls, prob in result["all_probs_%"].items():
        bar = "█" * int(prob / 5)
        mark = " [TERPILIH]" if cls == result["predicted"] else ""
        print(f"  {cls:25s}: {prob:6.2f}% {bar}{mark}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Penggunaan: python inference.py <nama_model> <path_gambar>")
        print(f"Pilihan model: {MODEL_LIST}")
        sys.exit(1)

    m_name = sys.argv[1]
    img_path = sys.argv[2]

    if m_name not in MODEL_LIST:
        print(f"Model tidak dikenal. Pilih salah satu dari: {MODEL_LIST}")
        sys.exit(1)

    try:
        model = load_model_for_inference(m_name)
        res = predict_single_image(model, img_path)
        print_prediction(res, display_name=m_name.upper())
    except Exception as e:
        print(f"Terjadi kesalahan saat inferensi: {str(e)}")