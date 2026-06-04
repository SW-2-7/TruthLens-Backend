from typing import Dict, Any, Optional

import torch
from PIL import Image

from .config import MODEL_LIST, DEFAULT_MODEL_NAME
from .model import create_model
from .preprocess import preprocess_pil


def _clean_state_dict(state: dict) -> dict:
    """DataParallel로 학습된 경우 key의 'module.' 접두사 제거."""
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """모델과 가중치를 로딩해서 eval 상태로 반환."""
    if model_name not in MODEL_LIST:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available: {list(MODEL_LIST.keys())}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MODEL_LIST[model_name]
    model = create_model(arch=cfg["arch"], num_classes=cfg["num_classes"])

    print(f"[INFO] Loading model: {cfg['weights']}")
    state = torch.load(cfg["weights"], map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(_clean_state_dict(state))
    model.to(device)
    model.eval()

    setattr(model, "threshold", float(cfg.get("threshold", 0.5)))
    setattr(model, "model_name", model_name)

    return model


@torch.no_grad()
def predict_from_pil(
    model: torch.nn.Module,
    img: Image.Image,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """PIL 이미지를 받아 fake 확률과 label 반환."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if threshold is None:
        threshold = float(getattr(model, "threshold", 0.5))

    x = preprocess_pil(img).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    fake_prob = float(probs[1].item())
    real_prob = float(probs[0].item())

    return {
        "label": "FAKE" if fake_prob >= threshold else "REAL",
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "threshold": threshold,
        "model_name": getattr(model, "model_name", None),
    }
