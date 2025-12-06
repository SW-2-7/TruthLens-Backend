from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.api import api_router
from app.model import load_model, DEFAULT_MODEL_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print(f"Loading model: {DEFAULT_MODEL_NAME}")
    app.state.model = load_model(DEFAULT_MODEL_NAME)
    app.state.device = next(app.state.model.parameters()).device
    print(f"Model loaded on device: {app.state.device}")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def health_check():
    return {"status": "ok", "msg": "서버 정상"}
