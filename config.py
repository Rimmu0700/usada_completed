"""
config.py
=========
Konfigurasi terpusat untuk seluruh project RESNET_USADA.
Ubah nilai di sini tanpa perlu menyentuh file lain.

REVISI MULTI-MODEL:
  - Mendukung 3 arsitektur: ResNet50, MobileNetV3-Small, ViT-Base/16
  - train.py akan otomatis melatih ketiganya secara berurutan
  - Setiap model punya folder output terpisah (outputs/<model_name>/)
"""

import torch
import os

# ==============================================================
# PATH DATASET
# ==============================================================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE     = os.path.join(BASE_DIR, "dataset_source")       # data mentah asli
DATA_SPLIT      = os.path.join(BASE_DIR, "dataset_split")        # hasil split
DATA_AUGMENTED  = os.path.join(BASE_DIR, "dataset_augmented")    # hasil augmentasi

TRAIN_DIR       = os.path.join(DATA_SPLIT, "train")
VAL_DIR         = os.path.join(DATA_SPLIT, "val")
TEST_DIR        = os.path.join(DATA_SPLIT, "test")
AUG_TRAIN_DIR   = os.path.join(DATA_AUGMENTED, "train")

# ==============================================================
# DAFTAR MODEL YANG DIBANDINGKAN
# ==============================================================
# [EXPERIMENT] train.py akan looping ke semua model di list ini secara berurutan.
# Untuk uji coba 1 model saja, kurangi isi list ini sementara.
MODEL_LIST = ["resnet50", "mobilenet", "vit"]

# Nama tampilan & info ringkas tiap model (untuk laporan/paper)
MODEL_INFO = {
    "resnet50":  {"display_name": "ResNet50",          "family": "CNN",         "params_class": "Heavy CNN"},
    "mobilenet": {"display_name": "MobileNetV3-Small",  "family": "CNN (lite)",  "params_class": "Lightweight CNN"},
    "vit":       {"display_name": "ViT-Base/16",        "family": "Transformer", "params_class": "Heavy Transformer"},
}

# ==============================================================
# PATH OUTPUT — DIPISAH PER MODEL
# ==============================================================
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")


def get_output_dirs(model_name: str) -> dict:
    """
    Mengembalikan semua path output untuk satu model tertentu.
    Dipanggil dari train.py / evaluate.py dengan model_name aktif.

    Hasil:
        outputs/<model_name>/logs/
        outputs/<model_name>/checkpoints/
        outputs/<model_name>/metrics/
        outputs/<model_name>/plots/
    """
    model_dir = os.path.join(OUTPUT_ROOT, model_name)
    dirs = {
        "model_dir":      model_dir,
        "log_dir":        os.path.join(model_dir, "logs"),
        "checkpoint_dir": os.path.join(model_dir, "checkpoints"),
        "metrics_dir":    os.path.join(model_dir, "metrics"),
        "plots_dir":      os.path.join(model_dir, "plots"),
    }
    dirs["best_model_path"]  = os.path.join(dirs["checkpoint_dir"], "best_model.pth")
    dirs["final_model_path"] = os.path.join(dirs["checkpoint_dir"], "final_model.pth")
    return dirs


# Path default (kompatibilitas dengan kode lama yang masih import langsung) —
# mengarah ke model PERTAMA di MODEL_LIST. Modul yang sudah di-upgrade
# (train.py, evaluate.py, inference.py) akan memanggil get_output_dirs()
# dengan model_name eksplisit, bukan memakai variabel default ini.
_default_dirs   = get_output_dirs(MODEL_LIST[0])
LOG_DIR         = _default_dirs["log_dir"]
CHECKPOINT_DIR  = _default_dirs["checkpoint_dir"]
METRICS_DIR     = _default_dirs["metrics_dir"]
PLOTS_DIR       = _default_dirs["plots_dir"]
BEST_MODEL_PATH  = _default_dirs["best_model_path"]
FINAL_MODEL_PATH = _default_dirs["final_model_path"]

# Path perbandingan gabungan 3 model (dibuat oleh compare.py)
COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "comparison")

# ==============================================================
# KELAS / LABEL SPESIES
# ==============================================================
# [EXPERIMENT] FASE 1 — uji coba pipeline dengan 5 spesies data terbanyak
CLASS_NAMES = [
    "Symphytum_officinale",      # 106 gambar — Boraginaceae
    "Euchresta_horsfieldii",     #  96 gambar — Lesch. & Benn.
    "Tabernaemontana_sp",        #  89 gambar
    "Zingiber_purpureum",        #  82 gambar — Roxb.
    "Erythrina_hypaphorus",      #  81 gambar — Boerl. ex Koord.
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# [EXPERIMENT] Jumlah gambar asli per kelas (urutan SAMA dengan CLASS_NAMES di atas).
# Dipakai untuk menghitung class_weights otomatis di train.py — TIDAK hardcode lagi.
CLASS_RAW_COUNTS = [106, 96, 89, 82, 81]

# ==============================================================
# IMAGE PROCESSING
# ==============================================================
# [PENTING] ViT-Base/16 dan ResNet50/MobileNet semua kompatibel dengan 224x224,
# jadi satu IMAGE_SIZE bisa dipakai untuk ketiganya tanpa konflik.
IMAGE_SIZE   = 224
CHANNELS     = 3
MEAN         = [0.485, 0.456, 0.406]   # ImageNet mean (standar utk ketiga model)
STD          = [0.229, 0.224, 0.225]   # ImageNet std

# ==============================================================
# SPLIT RATIO
# ==============================================================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ==============================================================
# AUGMENTASI — PROPORSIONAL PER KELAS
# ==============================================================
TARGET_TRAIN_PER_CLASS = 400
MAX_AUG_MULTIPLIER = 10
MIN_AUG_MULTIPLIER = 1

# ==============================================================
# IMBALANCE HANDLING
# ==============================================================
# [EXPERIMENT] Weighted loss dihitung OTOMATIS dari CLASS_RAW_COUNTS di train.py,
# tidak perlu hardcode angka manual lagi.
USE_CLASS_WEIGHTED_LOSS = True
USE_WEIGHTED_SAMPLER    = False

# ==============================================================
# TRAINING HYPERPARAMETER (DEFAULT — bisa di-override per model)
# ==============================================================
BATCH_SIZE      = 16
NUM_EPOCHS      = 40
LEARNING_RATE   = 0.0001
WEIGHT_DECAY    = 1e-4
MOMENTUM        = 0.9
NUM_WORKERS     = 4

LR_PATIENCE     = 3
LR_FACTOR       = 0.5
LR_MIN          = 1e-6

EARLY_STOP_PATIENCE = 10

# ==============================================================
# [EXPERIMENT] OVERRIDE HYPERPARAMETER PER MODEL
# ==============================================================
# ViT membutuhkan batch size lebih kecil (VRAM lebih besar per sampel)
# dan learning rate yang berbeda karena arsitektur attention sensitif terhadap lr besar.
# MobileNet lebih ringan, bisa pakai batch lebih besar jika VRAM cukup.
MODEL_HYPERPARAMS = {
    "resnet50": {
        "batch_size":    16,
        "learning_rate": 0.0001,
    },
    "mobilenet": {
        "batch_size":    32,       # lebih ringan → batch lebih besar aman di VRAM
        "learning_rate": 0.0005,   # MobileNet lebih toleran lr lebih tinggi
    },
    "vit": {
        "batch_size":    8,        # ViT-Base lebih berat di VRAM, turunkan batch
        "learning_rate": 0.00003,  # ViT fine-tuning butuh lr sangat kecil (lebih sensitif)
    },
}


def get_hyperparams(model_name: str) -> dict:
    """Ambil batch_size & learning_rate spesifik untuk satu model."""
    override = MODEL_HYPERPARAMS.get(model_name, {})
    return {
        "batch_size":    override.get("batch_size", BATCH_SIZE),
        "learning_rate": override.get("learning_rate", LEARNING_RATE),
    }


# ==============================================================
# GPU / CUDA
# ==============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False
CUDNN_BENCHMARK = True

# ==============================================================
# PRETRAINED / FREEZE SETTING
# ==============================================================
PRETRAINED      = True
FREEZE_BACKBONE = True   # backbone freeze di awal, gradual unfreeze di train.py

# ==============================================================
# LOGGING
# ==============================================================
LOG_INTERVAL = 10
SAVE_EVERY   = 5
