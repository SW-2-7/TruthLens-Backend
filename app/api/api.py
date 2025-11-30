from fastapi import APIRouter
from app.api.endpoints import detect

api_router = APIRouter()
api_router.include_router(detect.router, tags=["detect"])
