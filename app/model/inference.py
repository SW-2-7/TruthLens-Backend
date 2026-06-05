from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .config import ENSEMBLE_CONFIG, MODEL_LIST
from .face_detectors import MTCNNFaceDetector, crop_face
from .model import create_model
from .preprocess import preprocess_pil


def _clean_state_dict(state: dict) -> dict:
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


def _load_single_model(model_name: str, device: str) -> nn.Module:
    cfg = MODEL_LIST[model_name]
    weights_path = cfg["weights"]
    if not weights_path.exists():
        raise FileNotFoundError(f"가중치 파일 없음: {weights_path}")

    model = create_model(arch=cfg["arch"], num_classes=cfg["num_classes"])
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if isinstance(ckpt, dict):
        ckpt = _clean_state_dict(ckpt)

    model.load_state_dict(ckpt)
    return model.to(device).eval()


class EnsembleDetector:
    """앙상블 딥페이크 탐지기 (MTCNN 얼굴 감지 + 전체 이미지 폴백)."""

    def __init__(self, models: List[Dict[str, Any]], threshold: float, device: str, use_face_crop: bool = True) -> None:
        self.models = models  # [{"model": nn.Module, "weight": float, "name": str}]
        self.threshold = threshold
        self.device = device
        self.primary_model: nn.Module = models[0]["model"]  # Grad-CAM 용 (EfficientNet-B0)
        self.use_face_crop = use_face_crop
        self.face_detector: Optional[MTCNNFaceDetector] = None

        if use_face_crop:
            self._init_face_detector()

    def _init_face_detector(self) -> None:
        try:
            self.face_detector = MTCNNFaceDetector(device=self.device)
            print("[INFO] 얼굴 탐지기(MTCNN) 로드 완료")
        except Exception as e:
            print(f"[WARN] 얼굴 탐지기 로드 실패, 전체 이미지로 진행합니다: {e}")
            self.use_face_crop = False

    def parameters(self):
        """app.state.device 추출용."""
        return self.primary_model.parameters()

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        """
        얼굴 감지 후 앙상블 추론.
        반환값에 'analyzed_img' 포함 (Grad-CAM 생성에 사용).
        """
        img = img.convert("RGB")
        analyzed_img = img

        if self.use_face_crop and self.face_detector is not None:
            cropped = self._crop_face(img)
            if cropped is None:
                return {
                    "success": False,
                    "error": "no_face",
                    "message": "얼굴을 찾을 수 없습니다. 얼굴이 잘 보이는 사진을 올려주세요.",
                }
            analyzed_img = cropped

        score = self._ensemble_score(analyzed_img)
        label = "FAKE" if score >= self.threshold else "REAL"

        return {
            "success": True,
            "label": label,
            "fake_probability": score,
            "real_probability": round(1.0 - score, 4),
            "threshold": self.threshold,
            "analyzed_img": analyzed_img,
        }

    @torch.no_grad()
    def _ensemble_score(self, img: Image.Image) -> float:
        x = preprocess_pil(img).to(self.device)
        score = 0.0
        for item in self.models:
            prob = torch.softmax(item["model"](x), dim=1)[0, 1].item()
            score += item["weight"] * prob
        return round(score, 4)

    def _crop_face(self, img: Image.Image) -> Optional[Image.Image]:
        try:
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detection = self.face_detector.detect_main_face(img_bgr)
            if detection is None:
                return None
            face_bgr = crop_face(img_bgr, detection, output_size=224)
            if face_bgr is None or face_bgr.size == 0:
                return None
            return Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            return None


def load_model(device: Optional[str] = None) -> EnsembleDetector:
    """앙상블 모델 + 얼굴 탐지기 로딩."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded: List[Dict[str, Any]] = []
    for cfg in ENSEMBLE_CONFIG["models"]:
        model = _load_single_model(cfg["name"], device)
        loaded.append({"model": model, "weight": cfg["weight"], "name": cfg["name"]})
        print(f"[INFO]   ✓ {cfg['name']} (weight={cfg['weight']})")

    return EnsembleDetector(
        models=loaded,
        threshold=float(ENSEMBLE_CONFIG["threshold"]),
        device=device,
        use_face_crop=True,
    )


def predict_from_pil(detector: EnsembleDetector, img: Image.Image, **kwargs) -> Dict[str, Any]:
    """하위 호환성을 위한 래퍼."""
    return detector.predict(img)
