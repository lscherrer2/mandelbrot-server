## mandelbrot-server

A small FastAPI server that produces Mandelbrot Set PNGs. Rendering is done on the GPU/Apple 
Silicon (if available) with PyTorch. 

### Requirements

- Python >= 3.13
- `uv` package manager 

### Install

From the repo root:

```bash
./tool/install
```

This installs the package into a .venv in editable mode.

### Run the server (dev)

```bash
./tool/start
```

By default this starts Uvicorn with auto-reload on port `6477`.

### API

Server base URL (default): `http://127.0.0.1:6477`

#### GET /

Health check.

Example:

```bash
curl http://127.0.0.1:6477/
```

#### GET /mandelbrot/generate

Generates a Mandelbrot PNG and returns it `image/png`.

Example:

```bash
curl -o mandelbrot.png \
	"http://127.0.0.1:6477/mandelbrot/generate?width=1024&height=910&iter=250"
```

Query Parameters and Defaults:

- `xmin` / `xmax`: `-2.1` / `0.6`
- `ymin` / `ymax`: `-1.2` / `1.2`
- `iter`: `100`
- `width` / `height`: `1024` / `910`
- `initial_hue`: `0.0`
- `hue_shift_speed`: `1.0`
- `fade_saturation`: `false`
- `saturation_fade_rate`: `100.0`
- `fade_value`: `false`
- `value_fade_rate`: `100.0`

Example (zoom into a region):

```bash
curl -o zoom.png \
	"http://127.0.0.1:6477/mandelbrot/generate?xmin=-0.8&xmax=-0.7&ymin=0.05&ymax=0.15&iter=500&width=1200&height=900"
```

#### GET /mandelbrot/generate/stream

Streams a series of progressively computed Mandelbrot PNG frames.

The response is `application/octet-stream` and is a sequence of frames encoded as:

- 8-byte big-endian unsigned integer length
- followed by that many bytes of PNG data

Query Parameters (in addition to `/mandelbrot/generate` params):

- `render_frames`: number of frames to actually colorize/encode and emit (default: `10`)
- `render_from`: lower-bound iteration index to start emitting frames from (default: `1`)

Notes:

- The server still computes all iterations up to `iter`, but only emits `render_frames` frames.
- Frames are evenly spaced across iterations `[render_from, iter]` and always include the final image.


### Explorer UI

This server can power Mandelbrot UI: https://github.com/lscherrer2/mandelbrot-ui

- Instructions on setting up the UI are in the repo's README
