from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.schemas.detect import DetectResponse
from app.model.preprocess import preprocess_pil
from app.model.gradcam import GradCAM, get_target_layer, generate_heatmap_overlay
from PIL import Image
import io
import base64
import torch
import torch.nn.functional as F

router = APIRouter()

MAX_FILE_SIZE = 30 * 1024 * 1024
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
MAX_OUTPUT_PX = 1024


def validate_image_file(file: UploadFile, file_bytes: bytes) -> None:
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="파일 크기가 너무 큽니다. 최대 30MB까지 업로드 가능합니다."
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 파일 형식입니다. 이미지 파일(JPG, PNG, GIF, WEBP)만 업로드 가능합니다."
        )
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 파일 확장자입니다. 이미지 파일(JPG, PNG, GIF, WEBP)만 업로드 가능합니다."
            )


def _compress_to_base64(image: Image.Image, quality: int = 85) -> str:
    """최대 MAX_OUTPUT_PX 리사이즈 후 JPEG base64 인코딩."""
    w, h = image.size
    if max(w, h) > MAX_OUTPUT_PX:
        ratio = MAX_OUTPUT_PX / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _run_analysis(detector, img: Image.Image) -> dict:
    """얼굴 감지 + 앙상블 추론 + EfficientNet-B0 Grad-CAM."""
    # 1. 얼굴 감지 + 앙상블 점수
    result = detector.predict(img)

    if not result.get("success", True):
        return result  # no_face 에러 전파

    analyzed_img = result["analyzed_img"]
    target_class = 1 if result["label"] == "FAKE" else 0

    # 2. Grad-CAM은 primary 모델(EfficientNet-B0, weight 0.8)로 생성
    #    analyzed_img(얼굴 crop) 기준으로 히트맵 생성
    primary = detector.primary_model
    device = detector.device

    input_tensor = preprocess_pil(analyzed_img).to(device)
    input_tensor.requires_grad_(True)

    target_layer = get_target_layer(primary)
    gradcam = GradCAM(primary, target_layer)

    try:
        primary.eval()
        output = primary(input_tensor)
        primary.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        weights = gradcam.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * gradcam.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        gradcam.remove_hooks()

    return {
        "success": True,
        "label": result["label"],
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"],
        "heatmap_array": cam,
        "analyzed_img": analyzed_img,
    }


@router.post("/detect", response_model=DetectResponse)
async def detect_image(request: Request, file: UploadFile = File(...)):
    """
    Deepfake detection endpoint.
    - Max file size: 30MB
    - Allowed formats: JPG, PNG, GIF, WEBP
    - 얼굴이 감지되지 않으면 422 반환
    """
    image_bytes = await file.read()
    validate_image_file(file, image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="파일을 이미지로 열 수 없습니다. 유효한 이미지 파일인지 확인해주세요."
        )

    detector = request.app.state.model
    result = _run_analysis(detector, img)

    if not result.get("success", True):
        raise HTTPException(
            status_code=422,
            detail=result.get("message", "분석에 실패했습니다.")
        )

    # heatmap은 얼굴 crop 위에, original_image는 원본 전체 이미지
    heatmap_img = generate_heatmap_overlay(result["analyzed_img"], result["heatmap_array"])

    return DetectResponse(
        filename=file.filename or "unknown",
        is_fake=(result["label"] == "FAKE"),
        score=round(result["fake_probability"] * 100, 2),
        heatmap=_compress_to_base64(heatmap_img),
        original_image=_compress_to_base64(img),
    )
