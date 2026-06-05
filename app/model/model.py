# model/model.py

from torchvision import models
import torch.nn as nn


def create_model(arch: str = "efficientnet_b0", num_classes: int = 2, dropout_rate: float = 0.5) -> nn.Module:
    if arch == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model
