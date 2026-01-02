from typing import Literal
from collections.abc import Iterable, Generator
import cv2

import torch


@torch.no_grad()
def complex_plane(
    width: int,
    height: int,
    x: tuple[float, float],
    y: tuple[float, float],
    device: Literal["cpu", "cuda", "mps"],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_t: torch.Tensor = torch.linspace(
        min(x), max(x), width, device=device, dtype=torch.float32
    )
    y_t: torch.Tensor = torch.linspace(
        max(y), min(y), height, device=device, dtype=torch.float32
    )
    X, Y = torch.meshgrid(x_t, y_t, indexing="xy")
    return X, Y


@torch.no_grad()
def mandelbrot(
    width: int,
    height: int,
    x: tuple[float, float],
    y: tuple[float, float],
    iterations: int,
    device: Literal["cpu", "cuda", "mps"],
):
    res_grid = torch.full((height, width), -1, dtype=torch.int, device=device)

    c_re, c_im = complex_plane(width, height, x, y, device=device)

    z_re = c_re.clone()
    z_im = c_im.clone()

    for i in range(iterations):
        new_re = z_re * z_re - z_im * z_im + c_re
        new_im = 2.0 * z_re * z_im + c_im
        z_re, z_im = new_re, new_im

        mag2 = z_re * z_re + z_im * z_im
        mask = (res_grid == -1) & (mag2 > 4.0)
        res_grid[mask] = i

    return res_grid


@torch.no_grad()
def stream_mandelbrot(
    width: int,
    height: int,
    x: tuple[float, float],
    y: tuple[float, float],
    iterations: int,
    device: Literal["cpu", "cuda", "mps"],
) -> Generator[torch.Tensor]:
    res_grid = torch.full((height, width), -1, dtype=torch.int, device=device)

    c_re, c_im = complex_plane(width, height, x, y, device=device)

    z_re = c_re.clone()
    z_im = c_im.clone()

    for i in range(iterations):
        new_re = z_re * z_re - z_im * z_im + c_re
        new_im = 2.0 * z_re * z_im + c_im
        z_re, z_im = new_re, new_im

        mag2 = z_re * z_re + z_im * z_im
        mask = (res_grid == -1) & (mag2 > 4.0)
        res_grid[mask] = i

        yield res_grid


@torch.no_grad()
def to_color(
    mandelbrot: torch.Tensor,
    channels: Iterable[str] = "bgr",
    initial_hue: float = 0.0,
    hue_shift_speed: float = 1.0,
    fade_saturation: bool = False,
    saturation_fade_rate: float = 100.0,
    fade_value: bool = False,
    value_fade_rate: float = 100.0,
) -> torch.Tensor:
    mask_set = mandelbrot == -1
    cycle_length = 100.0 / max(hue_shift_speed, 0.01)  # Avoid division by zero

    h = ((mandelbrot.float() / cycle_length) + initial_hue) % 1.0
    h[mask_set] = 0

    s = torch.ones_like(h)
    if fade_saturation:
        s = torch.clamp(1.0 - (mandelbrot.float() / saturation_fade_rate), 0.0, 1.0)
    s[mask_set] = 0

    v = torch.ones_like(h)
    if fade_value:
        v = torch.clamp(1.0 - (mandelbrot.float() / value_fade_rate), 0.0, 1.0)
    v[mask_set] = 0

    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    h6 = h * 6
    mask0 = (h6 >= 0) & (h6 < 1)
    r[mask0] = c[mask0]
    g[mask0] = x[mask0]

    mask1 = (h6 >= 1) & (h6 < 2)
    r[mask1] = x[mask1]
    g[mask1] = c[mask1]

    mask2 = (h6 >= 2) & (h6 < 3)
    g[mask2] = c[mask2]
    b[mask2] = x[mask2]

    mask3 = (h6 >= 3) & (h6 < 4)
    g[mask3] = x[mask3]
    b[mask3] = c[mask3]

    mask4 = (h6 >= 4) & (h6 < 5)
    r[mask4] = x[mask4]
    b[mask4] = c[mask4]

    mask5 = (h6 >= 5) & (h6 < 6)
    r[mask5] = c[mask5]
    b[mask5] = x[mask5]

    r += m
    g += m
    b += m

    image = torch.stack([r, g, b], dim=-1) * 255
    image = image.to(torch.uint8)

    if channels == "rgb":
        pass
    elif channels == "bgr":
        image = image[:, :, [2, 1, 0]]

    return image


def generate_mandelbrot_image(
    width: int,
    height: int,
    x: tuple[float, float],
    y: tuple[float, float],
    iterations: int,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    channels: Iterable[str] = "bgr",
    initial_hue: float = 0.0,
    hue_shift_speed: float = 1.0,
    fade_saturation: bool = False,
    saturation_fade_rate: float = 100.0,
    fade_value: bool = False,
    value_fade_rate: float = 100.0,
) -> bytes:
    mandelbrot_set = mandelbrot(width, height, x, y, iterations, device)
    colorized = to_color(
        mandelbrot_set,
        channels=channels,
        initial_hue=initial_hue,
        hue_shift_speed=hue_shift_speed,
        fade_saturation=fade_saturation,
        saturation_fade_rate=saturation_fade_rate,
        fade_value=fade_value,
        value_fade_rate=value_fade_rate,
    )
    image_bytes = cv2.imencode(".png", colorized.cpu().numpy())[1].tobytes()
    return image_bytes


def stream_mandelbrot_images(
    width: int,
    height: int,
    x: tuple[float, float],
    y: tuple[float, float],
    iterations: int,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    channels: Iterable[str] = "bgr",
    initial_hue: float = 0.0,
    hue_shift_speed: float = 1.0,
    fade_saturation: bool = False,
    saturation_fade_rate: float = 100.0,
    fade_value: bool = False,
    value_fade_rate: float = 100.0,
    render_frames: int | None = None,
    render_from: int = 1,
) -> Generator[bytes]:
    if render_frames is not None and render_frames < 0:
        raise ValueError("render_frames must be >= 0 or None")

    if render_from < 1:
        raise ValueError("render_from must be >= 1")

    if iterations <= 0:
        return

    if render_from > iterations:
        for _ in stream_mandelbrot(width, height, x, y, iterations, device):
            pass
        return

    range_len = iterations - render_from + 1

    render_iterations: set[int] | None
    if render_frames is None:
        render_iterations = None
    else:
        frames = min(render_frames, range_len)
        if frames == 0:
            for _ in stream_mandelbrot(width, height, x, y, iterations, device):
                pass
            return

        if frames == 1:
            render_iterations = {iterations}
        else:
            render_iterations = {
                render_from + (i * (iterations - render_from)) // (frames - 1)
                for i in range(frames)
            }

    for i, mandelbrot_set in enumerate(
        stream_mandelbrot(width, height, x, y, iterations, device), start=1
    ):
        if i < render_from:
            continue
        if render_iterations is not None and i not in render_iterations:
            continue

        colorized = to_color(
            mandelbrot_set,
            channels=channels,
            initial_hue=initial_hue,
            hue_shift_speed=hue_shift_speed,
            fade_saturation=fade_saturation,
            saturation_fade_rate=saturation_fade_rate,
            fade_value=fade_value,
            value_fade_rate=value_fade_rate,
        )
        yield cv2.imencode(".png", colorized.cpu().numpy())[1].tobytes()
