import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import base64
import collections
from flask import Flask, request, jsonify, render_template
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

# --- Import YOLO dari ultralytics ---
from ultralytics import YOLO

# Import variabel YOLO dari config
from config import MODEL_LIST, DEVICE, CLASS_NAMES, NUM_CLASSES, get_output_dirs, YOLO_WEIGHTS_PATH, YOLO_CONF_THRESHOLD
from model import build_model

app = Flask(__name__)

# Memuat 3 model klasifikasi
loaded_models = {}
output_dirs = get_output_dirs(MODEL_LIST[0])  # Helper

print(f"Menggunakan Device: {DEVICE}")
for m_name in MODEL_LIST:
    dirs = get_output_dirs(m_name)
    checkpoint_path = dirs["best_model_path"]
    if os.path.exists(checkpoint_path):
        try:
            # FIX: build_model hanya menerima 1 argumen (model_name)
            model = build_model(m_name)
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

            # --- FIX 1: PENCOCOKAN KEY CHECKPOINT ---
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(DEVICE)
            model.eval()

            # --- FIX 2: PENCEGAHAN VRAM LEAK PADA SALIENCY MAP ---
            # Matikan gradien untuk semua bobot agar hemat memori saat inferensi
            for param in model.parameters():
                param.requires_grad = False

            loaded_models[m_name] = model
            print(f"Berhasil memuat model klasifikasi: {m_name}")
        except Exception as e:
            print(f"Gagal memuat model klasifikasi {m_name}: {str(e)}")
    else:
        print(f"File checkpoint tidak ditemukan untuk {m_name} di {checkpoint_path}")

# --- INISIALISASI MODEL YOLO11 DARI FOLDER WEIGHT (Via Config) ---
print(f"Memuat model YOLO11 dari: {YOLO_WEIGHTS_PATH}")
yolo_model = YOLO(YOLO_WEIGHTS_PATH)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def check_green_ratio(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    return green_pixels / total_pixels if total_pixels > 0 else 0.0


def draw_yolo_box(img_bgr, x1, y1, x2, y2, label="YOLO: Area Crop"):
    """Menggambar kotak bounding box merah + label pada gambar PENUH (bukan hasil crop)."""
    annotated = img_bgr.copy()
    color = (0, 0, 255)  # merah (format BGR)
    h, w = img_bgr.shape[:2]
    thickness = max(2, int(min(h, w) / 200))

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(h, w) / 900)
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
    label_y1 = max(0, y1 - text_h - 12)
    cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 12, y1), color, -1)
    cv2.putText(annotated, label, (x1 + 6, y1 - 6), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def generate_saliency_map(model, input_tensor, target_class, img_np):
    model.zero_grad()
    input_tensor.requires_grad_()

    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()

    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = cv2.resize(saliency, (img_np.shape[1], img_np.shape[0]))
    saliency = np.uint8(255 * saliency)

    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode('.jpg', overlay)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models/status")
def status():
    return jsonify({"loaded_models": list(loaded_models.keys())})


@app.route("/predict", methods=["POST"])
def predict():
    if not loaded_models:
        return jsonify({
            "global_status": {"accepted": False, "ood_reason": "Kesalahan: Belum ada model klasifikasi yang siap."},
            "models": {}
        }), 500

    data = request.get_json()
    b64_str = data["image"].split(",")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    original_img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # -----------------------------------------------------------------
    # DETEKSI AREA DAUN MENGGUNAKAN YOLO11
    # -> Menghasilkan 2 gambar terpisah:
    #    1. annotated_np : gambar PENUH + kotak merah (untuk ditampilkan, Tahap 1)
    #    2. img_np       : hasil CROP dekat area itu (untuk ditampilkan Tahap 2,
    #                       dan juga jadi input ke model klasifikasi)
    # -----------------------------------------------------------------
    yolo_results = yolo_model(original_img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = yolo_results[0].boxes

    annotated_np = original_img_np.copy()

    if len(boxes) > 0:
        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cropped_np = original_img_np[y1:y2, x1:x2]

        # --- FIX 3: PENCEGAHAN CRASH OPENCV (ZERO-DIMENSION) ---
        if cropped_np.size == 0:
            img_np = original_img_np.copy()
            yolo_status = "Daun terdeteksi tapi dimensi tidak valid. Menggunakan gambar penuh."
        else:
            img_np = cropped_np
            annotated_np = draw_yolo_box(original_img_np, x1, y1, x2, y2)
            yolo_status = f"Daun terdeteksi di koordinat ({x1}, {y1}) hingga ({x2}, {y2})."
    else:
        img_np = original_img_np.copy()
        yolo_status = "Daun tidak terdeteksi oleh YOLO11. Menggunakan gambar penuh sebagai fallback."

    # Gambar 1: gambar penuh + kotak merah (Tahap 1: Deteksi)
    _, annotated_buffer = cv2.imencode('.jpg', annotated_np)
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_buffer).decode('utf-8')}"

    # Gambar 2: hasil crop dekat (Tahap 2: Fokus Area) -> juga input klasifikasi
    _, crop_buffer = cv2.imencode('.jpg', img_np)
    cropped_b64 = f"data:image/jpeg;base64,{base64.b64encode(crop_buffer).decode('utf-8')}"
    # -----------------------------------------------------------------

    green_ratio = check_green_ratio(img_np)
    if green_ratio < 0.15:
        return jsonify({
            "global_status": {
                "accepted": False,
                "ood_reason": f"Gambar ditolak. Komponen warna hijau pada objek hanya {green_ratio*100:.1f}% (Minimal 15%)."
            },
            "annotated_image": annotated_b64,
            "cropped_image": cropped_b64,
            "yolo_status": yolo_status,
            "models": {}
        })

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    response_data = {
        "global_status": {},
        "models": {},
        "annotated_image": annotated_b64,
        "cropped_image": cropped_b64,
        "yolo_status": yolo_status
    }
    predictions = []
    max_confidences = []

    for m_name, model in loaded_models.items():
        t0 = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)

        inference_ms = (time.time() - t0) * 1000
        probs = F.softmax(outputs, dim=1)[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-6)).item()
        max_prob, predicted = torch.max(probs, 0)

        heatmap_b64 = None
        try:
            heatmap_b64 = generate_saliency_map(model, img_tensor, predicted.item(), img_np)
        except Exception:
            pass

        class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

        response_data["models"][m_name] = {
            "predicted": CLASS_NAMES[predicted.item()],
            "confidence": max_prob.item(),
            "entropy": entropy,
            "green_ratio": green_ratio,
            "inference_ms": round(inference_ms, 2),
            "class_probs": class_probs,
            "heatmap_b64": heatmap_b64
        }

        predictions.append(CLASS_NAMES[predicted.item()])
        max_confidences.append(max_prob.item())

    if predictions:
        avg_confidence = sum(max_confidences) / len(max_confidences)
        if avg_confidence < 0.25:
            response_data["global_status"] = {
                "accepted": False,
                "ood_reason": "Gambar ditolak. Rata-rata tingkat keyakinan model klasifikasi terlalu rendah."
            }
        else:
            response_data["global_status"] = {"accepted": True}
    else:
        response_data["global_status"] = {"accepted": False, "ood_reason": "Proses inferensi klasifikasi gagal."}

    return jsonify(response_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)