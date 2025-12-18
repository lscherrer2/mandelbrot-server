from fastapi import APIRouter
from mbserver.server.routes.generate import router as generate

__all__ = ["router"]

router = APIRouter(prefix="/mandelbrot")
router.include_router(generate)
