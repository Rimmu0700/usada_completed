import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights,
)
from config import NUM_CLASSES, DEVICE, PRETRAINED, FREEZE_BACKBONE

# Construct ResNet50 framework combined with modified linear classification tracking layers
def _build_resnet50() -> nn.Module:
    if PRETRAINED:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
        
    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

# Construct MobileNetV3-Small architecture combined with high-speed lightweight output heads
def _build_mobilenet() -> nn.Module:
    if PRETRAINED:
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v3_small(weights=None)
        
    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model

# Construct Vision Transformer model mapping patch elements to fully-connected outputs
def _build_vit() -> nn.Module:
    if PRETRAINED:
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = models.vit_b_16(weights=None)
        
    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    for param in model.heads.parameters():
        param.requires_grad = True
    return model

# Internal mapping registry connecting keys to internal construction targets
_BUILDERS = {
    "resnet50": _build_resnet50,
    "mobilenet": _build_mobilenet,
    "vit": _build_vit,
}

# Sequential module layout tracking to map deep backbone fine-tuning execution checkpoints
UNFREEZE_SCHEDULE = {
    "resnet50": ["layer4", "layer3", "layer2", "layer1"],
    "mobilenet": ["features.12", "features.11", "features.10", "features.9"],
    "vit": [
        "encoder.layers.encoder_layer_11",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_8"
    ],
}

# Global factory controller orchestrating structural allocation of target deep model states
def build_model(model_name: str) -> nn.Module:
    if model_name not in _BUILDERS:
        raise ValueError(f"Invalid model_name: {model_name}")
    model = _BUILDERS[model_name]()
    model = model.to(DEVICE)
    return model

# Unfreeze target layer blocks by enabling gradient updates matching sub-module substrings
def unfreeze_layer(model: nn.Module, layer_name: str):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True

# Retrieve unfreezing target list sequences assigned for structural model configurations
def get_unfreeze_schedule(model_name: str) -> list:
    return UNFREEZE_SCHEDULE.get(model_name, [])

# Summarize tracking parameters, checking relative weight counts and file memory footprints
def get_model_summary(model: nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    memory_mb = (total_params * 4) / (1024 ** 2)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable": non_trainable,
        "memory_mb": round(memory_mb, 2),
    }