# model/config.py

from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

MODEL_LIST: Dict[str, Dict[str, Any]] = {
    "eff_b0_finetuned": {
        "weights": WEIGHTS_DIR / "eff_b0_finetuned.pth",
        "arch": "efficientnet_b0",
        "num_classes": 2,
        "threshold": 0.5,
    },
    "mobilenet_v3_dfdc": {
        "weights": WEIGHTS_DIR / "dfdc_mobilenet_v3_focal.pth",
        "arch": "mobilenet_v3",
        "num_classes": 2,
        "threshold": 0.5,
    },
}

# EfficientNet-B0 파인튜닝(×0.8) + MobileNetV3-DFDC(×0.2)
ENSEMBLE_CONFIG: Dict[str, Any] = {
    "models": [
        {"name": "eff_b0_finetuned", "weight": 0.8},
        {"name": "mobilenet_v3_dfdc", "weight": 0.2},
    ],
    "threshold": 0.5,
}
