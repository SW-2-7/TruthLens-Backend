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


def _run_analysis(model: torch.nn.Module, img: Image.Image, device) -> dict:
    """추론 + GradCAM을 단일 forward/backward pass로 처리."""
    input_tensor = preprocess_pil(img).to(device)
    input_tensor.requires_grad_(True)
    threshold = float(getattr(model, "threshold", 0.5))

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    try:
        model.eval()

        # 단일 forward pass
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        fake_prob = float(probs[1].item())
        real_prob = float(probs[0].item())
        label = "FAKE" if fake_prob >= threshold else "REAL"
        target_class = 1 if label == "FAKE" else 0

        # GradCAM용 backward pass
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # 캡처된 gradient/activation으로 heatmap 계산
        weights = gradcam.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * gradcam.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    finally:
        gradcam.remove_hooks()

    return {
        "label": label,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "heatmap_array": cam,
    }


@router.post("/detect", response_model=DetectResponse)
async def detect_image(request: Request, file: UploadFile = File(...)):
    """
    Deepfake detection endpoint.
    - Max file size: 30MB
    - Allowed formats: JPG, PNG, GIF, WEBP
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

    model = request.app.state.model
    device = request.app.state.device

    result = _run_analysis(model, img, device)
    heatmap_img = generate_heatmap_overlay(img, result["heatmap_array"])

    return DetectResponse(
        filename=file.filename or "unknown",
        is_fake=(result["label"] == "FAKE"),
        score=round(result["fake_probability"] * 100, 2),
        heatmap=_compress_to_base64(heatmap_img),
        original_image=_compress_to_base64(img),
    )
