from mbserver.render import (
    generate_mandelbrot_image,
    stream_mandelbrot_images,
    determine_device,
)
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
import asyncio
from typing import Generator

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


def _create_stream_generator(
    width: int,
    height: int,
    x: tuple[float, float],
    y: tuple[float, float],
    iterations: int,
    device: str,
    initial_hue: float,
    hue_shift_speed: float,
    fade_saturation: bool,
    saturation_fade_rate: float,
    fade_value: bool,
    value_fade_rate: float,
    render_frames: int | None,
    render_from: int,
) -> Generator[bytes, None, None]:
    for image_bytes in stream_mandelbrot_images(
        width=width,
        height=height,
        x=x,
        y=y,
        iterations=iterations,
        device=device,
        initial_hue=initial_hue,
        hue_shift_speed=hue_shift_speed,
        fade_saturation=fade_saturation,
        saturation_fade_rate=saturation_fade_rate,
        fade_value=fade_value,
        value_fade_rate=value_fade_rate,
        render_frames=render_frames,
        render_from=render_from,
    ):
        length = len(image_bytes)
        yield length.to_bytes(8, byteorder="big") + image_bytes


async def _async_stream_wrapper(sync_generator: Generator[bytes, None, None]):
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    sentinel = object()

    def run_generator():
        try:
            for item in sync_generator:
                q.put(item)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=run_generator, daemon=True)
    thread.start()

    while True:
        item = await asyncio.to_thread(q.get)
        if item is sentinel:
            break
        yield item


@router.get("/stream")
async def generate_stream(
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
    render_frames: int | None = 10,
    render_from: int = 1,
):
    sync_gen = _create_stream_generator(
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
        render_frames=render_frames,
        render_from=render_from,
    )

    return StreamingResponse(
        _async_stream_wrapper(sync_gen),
        media_type="application/octet-stream",
    )
