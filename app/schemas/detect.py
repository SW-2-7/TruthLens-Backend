from pydantic import BaseModel

class DetectResponse(BaseModel):
    filename: str
    is_fake: bool
    score: float
    heatmap: str | None = None  # Base64 encoded string
    original_image: str | None = None  # Base64 encoded string
