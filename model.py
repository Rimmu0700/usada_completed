"""
model.py
========
Factory untuk membangun 3 arsitektur yang dibandingkan dalam penelitian:

  1. ResNet50           — CNN klasik dengan residual/skip connection
  2. MobileNetV3-Small   — CNN ringan, depthwise separable conv, dirancang untuk mobile/edge
  3. ViT-Base/16         — Vision Transformer, membagi image jadi patch lalu self-attention

KENAPA TIGA MODEL INI DIBANDINGKAN?
  - ResNet50   : representasi CNN "berat" standar, baseline kuat
  - MobileNetV3: representasi CNN "ringan", untuk lihat trade-off speed vs akurasi
  - ViT-Base/16: representasi arsitektur non-CNN (Transformer), beda total cara
                 memproses gambar — bukan sliding window konvolusi, tapi memotong
                 gambar jadi patch 16×16 lalu memperlakukannya seperti "kata"
                 dalam kalimat (self-attention).

ARSITEKTUR RINGKAS:

ResNet50 (Input 3×224×224):
  Conv1 → MaxPool → 4 stage Bottleneck (residual) → GAP → FC → NUM_CLASSES

MobileNetV3-Small (Input 3×224×224):
  Conv stem → blok inverted-residual dengan depthwise separable conv
  → Squeeze-and-Excitation → GAP → FC head → NUM_CLASSES
  Jauh lebih sedikit parameter & FLOPs dari ResNet50 — dirancang untuk
  perangkat dengan komputasi terbatas, relevan untuk perbandingan efisiensi GPU.

ViT-Base/16 (Input 3×224×224):
  Image dipecah jadi 14×14 = 196 patch (masing-masing 16×16 piksel)
  → setiap patch di-flatten & linear projection jadi "token" embedding
  → tambah [CLS] token + positional embedding
  → 12 layer Transformer Encoder (multi-head self-attention + MLP)
  → ambil representasi [CLS] token akhir → FC head → NUM_CLASSES
  Tidak ada konvolusi sama sekali — semua patch saling "melihat" satu sama
  lain lewat attention, beda fundamental dari cara CNN membaca gambar
  (yang membaca lokal lewat sliding window lalu menumpuk ke global).

GRADUAL UNFREEZE STRATEGY (dataset terbatas, semua model):
  Epoch awal  : hanya classifier head dilatih (backbone freeze total)
  Epoch 11+   : blok terdalam dibuka (mendekati output)
  Epoch 21+   : blok berikutnya dibuka
  Epoch 31+   : blok berikutnya lagi dibuka
  Blok paling awal (fitur paling generik) TETAP freeze.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights,
)

from config import NUM_CLASSES, DEVICE, PRETRAINED, FREEZE_BACKBONE


# ==============================================================
# 1. RESNET50
# ==============================================================

def _build_resnet50() -> nn.Module:
    """
    ResNet50 dengan FC head custom.
    Layer yang bisa di-gradual-unfreeze: layer4, layer3, layer2, layer1
    """
    if PRETRAINED:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)

    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
        # Softmax TIDAK disertakan — CrossEntropyLoss sudah menggabungkan
        # LogSoftmax + NLLLoss. Softmax hanya dipakai saat inferensi.
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# ==============================================================
# 2. MOBILENET V3 SMALL
# ==============================================================

def _build_mobilenet() -> nn.Module:
    """
    MobileNetV3-Small dengan classifier head custom.
    Layer yang bisa di-gradual-unfreeze: features.12, features.11, features.10 dst
    (MobileNetV3-Small punya 13 blok features, indeks 0-12)
    """
    if PRETRAINED:
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v3_small(weights=None)

    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False

    # MobileNetV3-Small classifier asli: Linear(576→1024) → Hardswish → Dropout → Linear(1024→1000)
    in_features = model.classifier[0].in_features   # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ==============================================================
# 3. VISION TRANSFORMER (ViT-Base/16)
# ==============================================================

def _build_vit() -> nn.Module:
    """
    ViT-Base/16 dengan classifier head custom.

    Struktur internal torchvision ViT:
      model.conv_proj          → patch embedding (Conv2d 16×16 stride 16, bukan konvolusi feature extractor biasa)
      model.encoder.layers.0-11 → 12 Transformer Encoder block
      model.heads               → classifier head ([CLS] token → output)

    Layer yang bisa di-gradual-unfreeze: encoder.layers.11, .10, .9 dst
    (blok terakhir dibuka duluan — paling dekat ke output, sama prinsipnya
    dengan layer4 di ResNet)
    """
    if PRETRAINED:
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = models.vit_b_16(weights=None)

    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False

    # ViT-Base/16 head asli: Linear(768 → 1000)
    in_features = model.heads.head.in_features   # 768
    model.heads = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    for param in model.heads.parameters():
        param.requires_grad = True

    return model


# ==============================================================
# FACTORY UTAMA
# ==============================================================

_BUILDERS = {
    "resnet50":  _build_resnet50,
    "mobilenet": _build_mobilenet,
    "vit":       _build_vit,
}

# Nama layer per model untuk gradual unfreeze, urut dari TERDALAM (dekat output)
# ke TERLUAR (dekat input). train.py akan membuka satu-per-satu sesuai urutan ini.
UNFREEZE_SCHEDULE = {
    "resnet50":  ["layer4", "layer3", "layer2", "layer1"],
    "mobilenet": ["features.12", "features.11", "features.10", "features.9"],
    "vit":       ["encoder.layers.encoder_layer_11",
                  "encoder.layers.encoder_layer_10",
                  "encoder.layers.encoder_layer_9",
                  "encoder.layers.encoder_layer_8"],
}


def build_model(model_name: str) -> nn.Module:
    """
    Bangun model sesuai nama yang diminta.

    Parameter:
        model_name : "resnet50" | "mobilenet" | "vit"

    Return:
        nn.Module yang sudah dipindah ke DEVICE (GPU/CPU)
    """
    if model_name not in _BUILDERS:
        raise ValueError(
            f"model_name tidak dikenal: '{model_name}'. "
            f"Pilihan valid: {list(_BUILDERS.keys())}"
        )

    print(f"\n[MODEL] Membangun '{model_name}' ...")
    model = _BUILDERS[model_name]()
    model = model.to(DEVICE)

    print(f"[MODEL] '{model_name}' siap di device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[GPU]   GPU: {torch.cuda.get_device_name(0)}")

    return model


def unfreeze_layer(model: nn.Module, layer_name: str):
    """
    Membuka satu blok/layer untuk dilatih (berlaku untuk model apapun,
    karena hanya mencocokkan substring nama parameter).

    Parameter:
        layer_name : substring nama layer, contoh "layer4", "features.12",
                     atau "encoder_layer_11"
    """
    count = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True
            count += 1
    print(f"[MODEL] Layer '{layer_name}' dibuka untuk training ({count} parameter tensors)")


def get_unfreeze_schedule(model_name: str) -> list:
    """Kembalikan urutan layer yang akan dibuka bertahap untuk model tertentu."""
    return UNFREEZE_SCHEDULE.get(model_name, [])


def get_model_summary(model: nn.Module) -> dict:
    """Hitung jumlah parameter model (total, trainable, frozen)."""
    total_params      = sum(p.numel() for p in model.parameters())
    trainable_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable     = total_params - trainable_params
    memory_mb = (total_params * 4) / (1024 ** 2)

    summary = {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "non_trainable":    non_trainable,
        "memory_mb":        round(memory_mb, 2),
    }

    print(f"\n[MODEL SUMMARY]")
    print(f"  Total parameters     : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters : {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters    : {non_trainable:,}")
    print(f"  Estimasi memory model: {memory_mb:.1f} MB")

    return summary


# ==============================================================
# QUICK TEST
# ==============================================================
if __name__ == "__main__":
    from config import DEVICE, MODEL_LIST

    for model_name in MODEL_LIST:
        print("\n" + "=" * 70)
        print(f"TEST MODEL: {model_name}")
        print("=" * 70)

        model = build_model(model_name)
        get_model_summary(model)

        dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            output = model(dummy)

        print(f"[FORWARD PASS TEST]")
        print(f"  Input shape  : {dummy.shape}")
        print(f"  Output shape : {output.shape}")

        probs = torch.softmax(output, dim=1)
        pred  = torch.argmax(probs, dim=1)
        print(f"  Prediksi kelas: {pred.tolist()}")

        # Test gradual unfreeze
        schedule = get_unfreeze_schedule(model_name)
        if schedule:
            unfreeze_layer(model, schedule[0])
            get_model_summary(model)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
