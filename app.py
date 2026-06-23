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
from config import MODEL_LIST, DEVICE, CLASS_NAMES, NUM_CLASSES, get_output_dirs
from model import build_model

app = Flask(__name__)

loaded_models = {}

for m_name in MODEL_LIST:
    path = get_output_dirs(m_name)["best_model_path"]
    if os.path.exists(path):
        m = build_model(m_name)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        if "model_state" in checkpoint:
            m.load_state_dict(checkpoint["model_state"])
        else:
            m.load_state_dict(checkpoint)
        m.eval()
        for param in m.parameters():
            param.requires_grad = False
        loaded_models[m_name] = m

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def check_green_ratio(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 30, 30])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    ratio = cv2.countNonZero(mask) / (img_np.shape[0] * img_np.shape[1])
    return ratio

def generate_saliency_map(model, img_tensor, predicted_class_idx, original_img_np):
    img_tensor_copy = img_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    with torch.enable_grad():
        outputs = model(img_tensor_copy)
        score = outputs[0][predicted_class_idx]
        score.backward()
    saliency, _ = torch.max(img_tensor_copy.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    superimposed_img = cv2.addWeighted(original_img_np, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models/status")
def status():
    return jsonify({"loaded_models": list(loaded_models.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    if not loaded_models:
        return jsonify({"global_status": {"accepted": False, "ood_reason": "Kesalahan: Belum ada model yang dilatih"}, "models": {}}), 500

    data = request.get_json()
    b64_str = data["image"].split(",")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    green_ratio = check_green_ratio(img_np)
    if green_ratio < 0.02:
        return jsonify({"global_status": {"accepted": False, "ood_reason": "Gambar ditolak. Komponen warna hijau terlalu rendah (< 2%)."}, "models": {}})

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    
    response_data = {"global_status": {}, "models": {}}
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
                "ood_reason": "Gambar ditolak. Tingkat keyakinan rata-rata klasifikasi terlalu rendah."
            }
        else:
            response_data["global_status"] = {
                "accepted": True
            }
    else:
        response_data["global_status"] = {"accepted": False, "ood_reason": "Inferensi gagal pada semua model."}

    return jsonify(response_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)