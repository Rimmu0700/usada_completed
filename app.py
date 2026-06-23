import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import base64
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

m_name = "mobilenet"
path = get_output_dirs(m_name)["best_model_path"]
if os.path.exists(path):
    m = build_model(m_name)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    m.eval()
    for param in m.parameters(): param.requires_grad = False
    loaded_models[m_name] = m

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/")
def index(): return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    b64_str = data["image"].split(",")[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_tensor = transform(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    
    response_data = {"global_status": {"accepted": True}, "models": {}}
    
    with torch.no_grad():
        outputs = loaded_models["mobilenet"](img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        max_prob, predicted = torch.max(probs, 0)
        
    response_data["models"]["mobilenet"] = {
        "predicted": CLASS_NAMES[predicted.item()],
        "confidence": max_prob.item(),
        "inference_ms": 0,
        "heatmap_b64": None
    }
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)