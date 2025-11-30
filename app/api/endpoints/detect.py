from fastapi import APIRouter, UploadFile, File
from app.schemas.detect import DetectResponse
import random

router = APIRouter()

@router.post("/detect", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Mock implementation of deepfake detection.
    Returns a random score and fake status.
    """
    # Simulate processing
    score = random.uniform(0, 100)
    is_fake = score > 50
    
    # In a real implementation, we would process the image here
    # and generate a heatmap.
    
    return DetectResponse(
        filename=file.filename,
        is_fake=is_fake,
        score=round(score, 2),
        heatmap="base64_encoded_heatmap_string_placeholder"
    )
