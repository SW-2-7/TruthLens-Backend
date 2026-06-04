from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "TruthLens Backend"
    API_V1_STR: str = "/api/v1"

    # 환경변수 예시: BACKEND_CORS_ORIGINS='["https://your-app.vercel.app"]'
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
