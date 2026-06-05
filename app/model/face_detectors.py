# model/face_detectors.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    landmarks: Optional[np.ndarray] = None  # shape (5, 2)


class MTCNNFaceDetector:
    def __init__(self, min_face_size: int = 40, device: str = "cpu"):
        try:
            from facenet_pytorch import MTCNN
        except ImportError as e:
            raise ImportError("facenet-pytorch가 설치되지 않았습니다: pip install facenet-pytorch") from e

        self.mtcnn = MTCNN(
            image_size=224,
            margin=0,
            min_face_size=min_face_size,
            post_process=False,
            keep_all=True,
            device=device,
        )

    def detect_main_face(self, img_bgr: np.ndarray) -> Optional[Detection]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return None

        probs = np.asarray(probs, dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32)
        landmarks = np.asarray(landmarks, dtype=np.float32) if landmarks is not None else None

        def rank(i: int) -> float:
            x1, y1, x2, y2 = boxes[i]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            score = float(probs[i]) if not math.isnan(float(probs[i])) else 0.0
            return area * score

        idx = max(range(len(boxes)), key=rank)
        x1, y1, x2, y2 = boxes[idx]
        x, y = int(round(x1)), int(round(y1))
        w, h = int(round(x2 - x1)), int(round(y2 - y1))
        lm = landmarks[idx] if landmarks is not None else None
        score = float(probs[idx]) if not math.isnan(float(probs[idx])) else 0.0
        return Detection(bbox=(x, y, w, h), score=score, landmarks=lm)


# ArcFace 5-point reference landmarks for 112×112 aligned face
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _align_face(img_bgr: np.ndarray, landmarks: np.ndarray, output_size: int) -> Optional[np.ndarray]:
    src = np.asarray(landmarks, dtype=np.float32)
    if src.shape != (5, 2):
        return None
    dst = _ARCFACE_REF_112 * (float(output_size) / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform is None:
        return None
    aligned = cv2.warpAffine(img_bgr, transform, (output_size, output_size), flags=cv2.INTER_LINEAR)
    return aligned if aligned is not None and aligned.size > 0 else None


def _expand_to_square(bbox: tuple, image_width: int, image_height: int, margin: float = 0.25) -> tuple:
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin)
    x1 = max(0, int(round(cx - side / 2.0)))
    y1 = max(0, int(round(cy - side / 2.0)))
    x2 = min(image_width, int(round(cx + side / 2.0)))
    y2 = min(image_height, int(round(cy + side / 2.0)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def crop_face(img_bgr: np.ndarray, detection: Detection, output_size: int = 224, margin: float = 0.25) -> Optional[np.ndarray]:
    """얼굴 정렬 crop. 랜드마크가 있으면 ArcFace 정렬, 없으면 bbox crop."""
    if detection.landmarks is not None:
        aligned = _align_face(img_bgr, detection.landmarks, output_size)
        if aligned is not None:
            return aligned

    image_height, image_width = img_bgr.shape[:2]
    x, y, w, h = _expand_to_square(detection.bbox, image_width, image_height, margin)
    if w <= 0 or h <= 0:
        return None
    face = img_bgr[y:y + h, x:x + w]
    if face.size == 0:
        return None
    return cv2.resize(face, (output_size, output_size), interpolation=cv2.INTER_AREA)
