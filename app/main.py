from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.api import api_router
from app.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ensemble model...")
    detector = load_model()
    app.state.model = detector
    app.state.device = torch.device(detector.device)
    print(f"Ensemble model ready on device: {detector.device}")
    yield
    print("Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

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
