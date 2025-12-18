from mbserver.render import generate_mandelbrot_image, determine_device
from fastapi import APIRouter, Response
import asyncio

router = APIRouter(prefix="/generate")

__all__ = ["router"]

DEVICE = determine_device()


@router.get("")
async def generate(
    xmin: float = -2.1,
    xmax: float = 0.6,
    ymin: float = -1.2,
    ymax: float = 1.2,
    iter: int = 100,
    width: int = 1024,
    height: int = 910,
    initial_hue: float = 0.0,
    hue_shift_speed: float = 1.0,
    fade_saturation: bool = False,
    saturation_fade_rate: float = 100.0,
    fade_value: bool = False,
    value_fade_rate: float = 100.0,
):
    buf = await asyncio.to_thread(
        generate_mandelbrot_image,
        width=width,
        height=height,
        x=(xmin, xmax),
        y=(ymin, ymax),
        iterations=iter,
        device=DEVICE,
        initial_hue=initial_hue,
        hue_shift_speed=hue_shift_speed,
        fade_saturation=fade_saturation,
        saturation_fade_rate=saturation_fade_rate,
        fade_value=fade_value,
        value_fade_rate=value_fade_rate,
    )
    return Response(content=buf, media_type="image/png")
