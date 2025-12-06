from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.schemas.detect import DetectResponse
from app.model import predict_from_pil, generate_gradcam_base64
from PIL import Image
import io

router = APIRouter()

# Constants
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp"]


def validate_image_file(file: UploadFile, file_bytes: bytes) -> None:
    """
    Validate uploaded file is an image and within size limit.
    Raises HTTPException if validation fails.
    """
    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기가 너무 큽니다. 최대 30MB까지 업로드 가능합니다."
        )
    
    # Check content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 이미지 파일(JPG, PNG, GIF, WEBP)만 업로드 가능합니다."
        )
    
    # Check file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 확장자입니다. 이미지 파일(JPG, PNG, GIF, WEBP)만 업로드 가능합니다."
            )


@router.post("/detect", response_model=DetectResponse)
async def detect_image(request: Request, file: UploadFile = File(...)):
    """
    Deepfake detection endpoint.
    Receives an image and returns prediction results with Grad-CAM heatmap.
    
    - Max file size: 30MB
    - Allowed formats: JPG, PNG, GIF, WEBP
    """
    # Read image bytes
    image_bytes = await file.read()
    
    # Validate file
    validate_image_file(file, image_bytes)
    
    # Try to open as image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="파일을 이미지로 열 수 없습니다. 유효한 이미지 파일인지 확인해주세요."
        )
    
    # Get model from app state
    model = request.app.state.model
    device = request.app.state.device
    
    # Run inference
    result = predict_from_pil(model, img, device=device)
    
    # Generate Grad-CAM heatmap
    heatmap_base64 = generate_gradcam_base64(model, img, device=device)
    
    # Map model output to API response
    return DetectResponse(
        filename=file.filename,
        is_fake=(result["label"] == "FAKE"),
        score=round(result["fake_probability"] * 100, 2),
        heatmap=heatmap_base64
    )
