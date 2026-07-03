import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE = os.path.join(BASE_DIR, "dataset_source")
DATA_SPLIT = os.path.join(BASE_DIR, "dataset_split")
DATA_AUGMENTED = os.path.join(BASE_DIR, "dataset_augmented")

TRAIN_DIR = os.path.join(DATA_SPLIT, "train")
VAL_DIR = os.path.join(DATA_SPLIT, "val")
TEST_DIR = os.path.join(DATA_SPLIT, "test")
AUG_TRAIN_DIR = os.path.join(DATA_AUGMENTED, "train")

MODEL_LIST = ["resnet50", "mobilenet", "vit"]

MODEL_INFO = {
    "resnet50":  {"display_name": "ResNet50",          "family": "CNN",         "params_class": "Heavy CNN"},
    "mobilenet": {"display_name": "MobileNetV3-Small", "family": "CNN (lite)",  "params_class": "Lightweight CNN"},
    "vit":       {"display_name": "ViT-Base/16",       "family": "Transformer", "params_class": "Heavy Transformer"},
}

OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")

def get_output_dirs(model_name: str) -> dict:
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

_default_dirs = get_output_dirs(MODEL_LIST[0])
LOG_DIR = _default_dirs["log_dir"]
CHECKPOINT_DIR = _default_dirs["checkpoint_dir"]
METRICS_DIR = _default_dirs["metrics_dir"]
PLOTS_DIR = _default_dirs["plots_dir"]
BEST_MODEL_PATH = _default_dirs["best_model_path"]
FINAL_MODEL_PATH = _default_dirs["final_model_path"]

COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "comparison")

if os.path.exists(DATA_SOURCE):
    CLASS_NAMES = sorted([d for d in os.listdir(DATA_SOURCE) if os.path.isdir(os.path.join(DATA_SOURCE, d)) and not d.startswith('.')])
    CLASS_RAW_COUNTS = [len([f for f in os.listdir(os.path.join(DATA_SOURCE, c)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) for c in CLASS_NAMES]
else:
    CLASS_NAMES = [
        "Alpinia galanga(L) willd.(angiberaceae)",
        "Antidesma bunius (L.) Spreng.",
        "Blumea balsamifera (L.) DC.",
        "Cinnamomum verum J.Presl",
        "Curcuma sylvatica vahl (Zingiberaceae)",
        "Erythrina hypaphorus Boerl. ex Koord.",
        "Euchresta horsfieldii (Lesch.) Benn.",
        "Graptophyllum pictum (L.) Griff.",
        "Justicia gendarussa Burm.f.",
        "Piper betle L.",
        "Symphytum officinale L.(boraginaceae)",
        "Tabernaemontana sp.",
        "Zingiber Purpureum Roxb",
        "Zingiber officinale Roxb",
        "amomum compactum sol.ex manton",
        "anredera cordifolia (ten) steenis",
        "plantago major L(plantaginaceae)"
    ]
    CLASS_NAMES = sorted(CLASS_NAMES)
    CLASS_RAW_COUNTS = [400] * len(CLASS_NAMES)

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMAGE_SIZE = 224
CHANNELS = 3
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

TARGET_TRAIN_PER_CLASS = 400
MAX_AUG_MULTIPLIER = 10
MIN_AUG_MULTIPLIER = 1

USE_CLASS_WEIGHTED_LOSS = True
USE_WEIGHTED_SAMPLER = False

BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
NUM_WORKERS = 4

LR_PATIENCE = 3
LR_FACTOR = 0.5
LR_MIN = 1e-6

EARLY_STOP_PATIENCE = 20
LR_PATIENCE         = 7

MODEL_HYPERPARAMS = {
    "resnet50": {
        "batch_size": 8,
        "learning_rate": 0.0001,
    },
    "mobilenet": {
        "batch_size": 16,
        "learning_rate": 0.0005,
    },
    "vit": {
        "batch_size": 4,
        "learning_rate": 0.00003,
    },
}

def get_hyperparams(model_name: str) -> dict:
    override = MODEL_HYPERPARAMS.get(model_name, {})
    return {
        "batch_size": override.get("batch_size", BATCH_SIZE),
        "learning_rate": override.get("learning_rate", LEARNING_RATE),
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False
CUDNN_BENCHMARK = True

PRETRAINED = True
FREEZE_BACKBONE = True

LOG_INTERVAL = 10
SAVE_EVERY = 5