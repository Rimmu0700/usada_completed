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

# --- Import YOLO from ultralytics ---
from ultralytics import YOLO

# Import YOLO variables from config
from config import MODEL_LIST, DEVICE, CLASS_NAMES, NUM_CLASSES, get_output_dirs, YOLO_WEIGHTS_PATH, YOLO_CONF_THRESHOLD
from model import build_model

app = Flask(__name__)


# =====================================================================
# GRAD-CAM (replaces the old vanilla saliency map)
#
# Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
# Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via
# Gradient-Based Localization. IEEE ICCV, 618-626.
#
# Works for CNN backbones (ResNet50, MobileNetV3-Small) directly, and for
# ViT-Base/16 via vit_reshape_transform, which turns its token embeddings
# into a spatial grid so the same CAM math applies to all three models.
#
# Safe to use even though every model.parameters() below gets
# requires_grad=False for memory savings — __call__() makes the INPUT
# tensor require grad instead, the same way the old saliency function did,
# so gradients still reach the hooked layer regardless of frozen weights.
# =====================================================================

class GradCAM:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.reshape_transform = reshape_transform
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations = activation.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = grad.detach()

    def __call__(self, input_tensor, target_class):
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_()

        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def vit_reshape_transform(tensor, grid_size=14):
    """(B, N+1, C) token embeddings -> (B, C, H, W) grid, dropping the CLS
    token. grid_size = image_size / patch_size = 224/16 = 14 for the 224x224
    input used in `transform` below — change if config.py trains at a
    different resolution."""
    patch_tokens = tensor[:, 1:, :]
    b, n, c = patch_tokens.shape
    result = patch_tokens.reshape(b, grid_size, grid_size, c)
    return result.permute(0, 3, 1, 2)


def infer_arch_name(m_name):
    """Maps a MODEL_LIST entry to one of the three architecture keys below,
    by substring match (case-insensitive). Edit directly if your MODEL_LIST
    strings don't contain these keywords."""
    name = m_name.lower()
    if "resnet" in name:
        return "resnet50"
    elif "mobilenet" in name:
        return "mobilenetv3_small"
    elif "vit" in name:
        return "vit_base_16"
    raise ValueError(f"Can't infer architecture from model name '{m_name}'")


def get_gradcam(model, arch_name):
    """Target layers assume standard torchvision structure. Replacing only
    the classifier head in build_model() should NOT affect these paths —
    that happens after these layers. print(model) once if any architecture
    fails to hook correctly."""
    if arch_name == "resnet50":
        return GradCAM(model, target_layer=model.layer4[-1])
    elif arch_name == "mobilenetv3_small":
        return GradCAM(model, target_layer=model.features[-1])
    elif arch_name == "vit_base_16":
        return GradCAM(
            model,
            target_layer=model.encoder.layers[-1],
            reshape_transform=vit_reshape_transform,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name!r}")


def generate_gradcam_map(gradcam, input_tensor, target_class, img_bgr):
    """Drop-in replacement for the old generate_saliency_map(). Same base64
    JPEG return format and same BGR overlay convention — only the call
    site inside /predict needed to change."""
    cam = gradcam(input_tensor, target_class)
    cam_resized = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.jpg', overlay)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


# =====================================================================
# Load 3 classification models
# =====================================================================
loaded_models = {}
output_dirs = get_output_dirs(MODEL_LIST[0])  # Helper

print(f"Using Device: {DEVICE}")
for m_name in MODEL_LIST:
    dirs = get_output_dirs(m_name)
    checkpoint_path = dirs["best_model_path"]
    if os.path.exists(checkpoint_path):
        try:
            # FIX: build_model only accepts 1 argument (model_name)
            model = build_model(m_name)
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

            # --- FIX 1: CHECKPOINT KEY MATCHING ---
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(DEVICE)
            model.eval()

            # --- FIX 2: PREVENTION OF VRAM LEAK ON SALIENCY MAP ---
            # Disable gradients for all weights to save memory during inference
            for param in model.parameters():
                param.requires_grad = False

            loaded_models[m_name] = model
            print(f"Successfully loaded classification model: {m_name}")
        except Exception as e:
            print(f"Failed to load classification model {m_name}: {str(e)}")
    else:
        print(f"Checkpoint file not found for {m_name} at {checkpoint_path}")

# --- Build one Grad-CAM wrapper per loaded model. Hooks are registered
#     once here, NOT inside /predict — registering them per-request would
#     accumulate hooks on every prediction and eventually leak memory /
#     corrupt results.
loaded_gradcams = {}
for m_name, model in loaded_models.items():
    try:
        loaded_gradcams[m_name] = get_gradcam(model, infer_arch_name(m_name))
        print(f"Grad-CAM ready for: {m_name}")
    except Exception as e:
        print(f"Grad-CAM setup failed for {m_name}: {str(e)} (heatmap will be skipped for this model)")

# --- INITIALIZE YOLO11 MODEL FROM WEIGHT FOLDER (Via Config) ---
print(f"Loading YOLO11 model from: {YOLO_WEIGHTS_PATH}")
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
    """Draw red bounding box + label on the FULL image (not the cropped result)."""
    annotated = img_bgr.copy()
    color = (0, 0, 255)  # red (BGR format)
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
            "global_status": {"accepted": False, "ood_reason": "Error: No classification models are ready."},
            "models": {}
        }), 500

    data = request.get_json()
    b64_str = data["image"].split(",")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    original_img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # -----------------------------------------------------------------
    # DETECT LEAF AREA USING YOLO11
    # -> Generates 2 separate images:
    #    1. annotated_np : FULL image + red box (for display, Stage 1)
    #    2. img_np       : CROPPED result near that area (for display Stage 2,
    #                       and also used as input for the classification model)
    # -----------------------------------------------------------------
    yolo_results = yolo_model(original_img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = yolo_results[0].boxes

    annotated_np = original_img_np.copy()

    if len(boxes) > 0:
        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cropped_np = original_img_np[y1:y2, x1:x2]

        # --- FIX 3: PREVENT OPENCV CRASH (ZERO-DIMENSION) ---
        if cropped_np.size == 0:
            img_np = original_img_np.copy()
            yolo_status = "Leaf detected but dimensions are invalid. Using full image."
        else:
            img_np = cropped_np
            annotated_np = draw_yolo_box(original_img_np, x1, y1, x2, y2)
            yolo_status = f"Leaf detected at coordinates ({x1}, {y1}) to ({x2}, {y2})."
    else:
        img_np = original_img_np.copy()
        yolo_status = "Leaf not detected by YOLO11. Using full image as fallback."

    # Image 1: full image + red box (Stage 1: Detection)
    _, annotated_buffer = cv2.imencode('.jpg', annotated_np)
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(annotated_buffer).decode('utf-8')}"

    # Image 2: close-up crop result (Stage 2: Area Focus) -> also classification input
    _, crop_buffer = cv2.imencode('.jpg', img_np)
    cropped_b64 = f"data:image/jpeg;base64,{base64.b64encode(crop_buffer).decode('utf-8')}"
    # -----------------------------------------------------------------

    green_ratio = check_green_ratio(img_np)
    if green_ratio < 0.15:
        return jsonify({
            "global_status": {
                "accepted": False,
                "ood_reason": f"Image rejected. Green color component on the object is only {green_ratio*100:.1f}% (Minimum 15%)."
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
        if m_name in loaded_gradcams:
            try:
                heatmap_b64 = generate_gradcam_map(
                    loaded_gradcams[m_name], img_tensor, predicted.item(), img_np
                )
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
                "ood_reason": "Image rejected. Average classification model confidence is too low."
            }
        else:
            response_data["global_status"] = {"accepted": True}
    else:
        response_data["global_status"] = {"accepted": False, "ood_reason": "Classification inference process failed."}

    return jsonify(response_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)