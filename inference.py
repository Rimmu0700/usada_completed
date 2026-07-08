import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from ultralytics import YOLO
from config import DEVICE, CLASS_NAMES, IDX_TO_CLASS, IMAGE_SIZE, MEAN, STD, MODEL_LIST, get_output_dirs, YOLO_WEIGHTS_PATH, YOLO_CONF_THRESHOLD
from model import build_model

# Load the pretrained YOLO detection engine
yolo_model = YOLO(YOLO_WEIGHTS_PATH)

# Define inference-time image transformation sequence
inference_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# Initialize the classification model and restore weights from checkpoints
def load_model_for_inference(model_name: str) -> nn.Module:
    dirs = get_output_dirs(model_name)
    checkpoint_path = dirs["best_model_path"]
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint untuk {model_name} tidak ditemukan di: {checkpoint_path}")
        
    num_classes = len(CLASS_NAMES)
    model = build_model(model_name, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # --- FIX 1: PENCOCOKAN KEY CHECKPOINT ---
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

# Run end-to-end inference including detection and classification
def predict_single_image(model: nn.Module, image_path: str) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    # Run YOLO detection for auto-cropping
    img_np = np.array(img)
    yolo_results = yolo_model(img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)
    boxes = yolo_results[0].boxes

    if len(boxes) > 0:
        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        cropped_img_np = img_np[y1:y2, x1:x2]
        
        # --- FIX 3: AMANKAN CROP ---
        if cropped_img_np.size == 0:
            img = Image.fromarray(img_np)
            yolo_status = "Dimensi daun invalid (Menggunakan Gambar Penuh)"
        else:
            img = Image.fromarray(cropped_img_np)
            yolo_status = f"Terdeteksi Daun (Crop: {img.size[0]}x{img.size[1]})"
    else:
        yolo_status = "Leaf not detected by YOLO. Defaulting to full image."

    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

    # Perform inference and record timing information
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
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

    probs = torch.softmax(logits, dim=1).squeeze(0)
    confidence = probs.max().item()
    pred_idx = probs.argmax().item()
    pred_class = IDX_TO_CLASS[pred_idx]

    # Map output distribution to class probabilities
    all_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(CLASS_NAMES))}
    all_probs_sorted = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return {
        "image_path": image_path,
        "original_size": original_size,
        "input_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "predicted": pred_class,
        "confidence": round(confidence * 100, 2),
        "inference_ms": round(inference_ms, 3),
        "all_probs_percent": all_probs_sorted,
        "yolo_status": yolo_status
    }

# Display prediction output cleanly on the terminal
def print_prediction(result: dict, display_name: str = ""):
    print("\n======================================================================")
    title = f"SYSTEM PREDICTION RESULTS: {display_name}" if display_name else "MEDICAL PLANT PREDICTION RESULTS"
    print(title)
    print("======================================================================")
    print(f"  File Name         : {os.path.basename(result['image_path'])}")
    print(f"  Original Size     : {result['original_size'][0]}x{result['original_size'][1]} px")
    print(f"  YOLO11 System     : {result['yolo_status']}")
    print(f"  Classifier Size   : {result['input_size']}")
    print(f"  Predicted Class   : {result['predicted']}")
    print(f"  Confidence Level  : {result['confidence']:.2f}%")
    print(f"  Processing Time   : {result['inference_ms']:.3f} ms")
    print("----------------------------------------------------------------------")
    print("Class Distribution Probability:")
    for cls, prob in result["all_probs_percent"].items():
        bar_count = int(prob / 5)
        bar = "".join(["*" for _ in range(bar_count)])
        mark = " [SELECTED]" if cls == result["predicted"] else ""
        print(f"  {cls:35s}: {prob:6.2f}% {bar}{mark}")
    print("======================================================================")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python inference.py <model_name> <image_path>")
        print(f"Available models: {MODEL_LIST}")
        sys.exit(1)

    m_name = sys.argv[1]
    img_path = sys.argv[2]

    if m_name not in MODEL_LIST:
        print(f"[ERROR] Unknown model. Select one from {MODEL_LIST}")
        sys.exit(1)

    try:
        model = load_model_for_inference(m_name)
        res = predict_single_image(model, img_path)
        print_prediction(res, display_name=m_name.upper())
    except Exception as e:
        print(f"[ERROR] Exception occurred during inference: {str(e)}")