from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "TruthLens Backend"
    API_V1_STR: str = "/api/v1"
    
    # CORS Configuration
    # In production, this should be a list of allowed origins
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        case_sensitive = True

settings = Settings()
