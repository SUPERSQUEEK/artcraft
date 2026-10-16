"""On-device image server for ArtCraft's "Local" provider (Qwen-Image 2.1 Turbo on Apple Silicon).

ArtCraft enqueues a job, polls it, then downloads the finished PNGs over HTTP:

    POST /v1/generate   {"model", "prompt", "width", "height", "num_images", "images"} -> {"job_id"}
    GET  /v1/jobs/<id>  -> {"status": queued|running|completed|failed, "images": [urls], "error"}
    GET  /files/<name>  -> image/png
    GET  /health        -> {"ok": true, ...}

`images` (optional) are reference images to edit, as http(s) URLs or base64 bytes; with
them and no width/height, the output follows the last image's aspect ratio.

Jobs run one at a time. The BF16 weights (17.5 GB text encoder + 14.2 GB transformer)
don't fit in 36 GB unified memory together, so the transformer + VAE stay resident and
the text encoder is loaded per job, used, and freed before denoising.
"""

import errno
import base64
import gc
import io
import json
import logging
import os
import queue
import random
import re
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from PIL import Image

HOST = os.environ.get("LOCAL_IMAGE_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCAL_IMAGE_SERVER_PORT", "8189"))
OUTPUT_DIR = Path(os.environ.get("LOCAL_IMAGE_SERVER_OUTPUT", Path.home() / "Pictures" / "ArtCraft Local"))
IDLE_UNLOAD_SECS = int(os.environ.get("LOCAL_IMAGE_SERVER_IDLE_UNLOAD_SECS", "900"))

MODELS = {"qwen_image_2p1_turbo": "Qwen/Qwen-Image-2.1-Turbo"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_SIDE = 2048
MAX_IMAGES = 4
MAX_REFERENCE_IMAGES = 3
MAX_REFERENCE_BYTES = 30 * 1024 * 1024
# Matches the pipeline's `output_resolution` default; condition images are resized to this area.
OUTPUT_RESOLUTION = 1024
USER_AGENT = "ArtCraft-LocalImageServer/1.0"
FILE_NAME_RE = re.compile(r"^local_[0-9a-f]{32}_\d+\.png$")

log = logging.getLogger("local_image_server")


class Job:
  def __init__(self, request):
    self.id = f"local_{uuid.uuid4().hex}"
    self.request = request
    self.status = "queued"
    self.images = []
    self.error = None


class Engine:
  """Owns the model; only the worker thread touches it."""

  def __init__(self):
    self.repo = None
    self.pipe = None
    self.last_used = 0.0

  def generate(self, job):
    req = job.request
    repo = MODELS[req["model"]]
    references = [load_image(source) for source in req["images"]]
    t0 = time.time()
    embeds, mask, pad_mask = self._encode(repo, req["prompt"], references)
    log.info("encoded prompt (+%d reference images) in %.1fs", len(references), time.time() - t0)

    pipe = self._denoiser(repo)
    base_seed = random.randrange(2**31)
    for i in range(req["num_images"]):
      t1 = time.time()
      image = self._denoise(pipe, embeds, mask, pad_mask, references, req, base_seed + i)
      name = f"{job.id}_{i}.png"
      image.save(OUTPUT_DIR / name)
      job.images.append(f"http://{HOST}:{PORT}/files/{name}")
      log.info("image %d/%d in %.1fs", i + 1, req["num_images"], time.time() - t1)
    self.last_used = time.time()

  def maybe_unload_idle(self):
    if self.pipe is not None and time.time() - self.last_used > IDLE_UNLOAD_SECS:
      log.info("idle for %ds, unloading model", IDLE_UNLOAD_SECS)
      self.pipe = self.repo = None
      free_memory()

  def _encode(self, repo, prompt, references):
    enc = QwenImage21Pipeline.from_pretrained(repo, transformer=None, vae=None, dtype=DTYPE).to(DEVICE)
    try:
      with torch.no_grad():
        embeds, mask, pad_mask = enc.encode_prompt(
          prompt, image=encoder_images(enc, references) or None, device=DEVICE
        )
      embeds = embeds.detach().clone()
      mask = mask.detach().clone() if mask is not None else None
      pad_mask = pad_mask.detach().clone()
    finally:
      del enc
      free_memory()
    return embeds, mask, pad_mask

  def _denoise(self, pipe, embeds, mask, pad_mask, references, req, seed):
    kwargs = dict(
      prompt_embeds=embeds,
      prompt_embeds_mask=mask,
      width=req["width"],
      height=req["height"],
      use_kv_cache=True,
      generator=torch.Generator("cpu").manual_seed(seed),
    )
    if not references:
      return pipe(**kwargs).images[0]
    # The pipeline can't take precomputed embeddings together with condition images (it has no
    # `image_pad_mask` argument), so hand back the encoding we already made instead of re-encoding.
    pipe.encode_prompt = lambda *args, **kw: (embeds, mask, pad_mask)
    try:
      return pipe(image=references, **kwargs).images[0]
    finally:
      del pipe.encode_prompt

  def _denoiser(self, repo):
    if self.pipe is None or self.repo != repo:
      self.pipe = None
      free_memory()
      t0 = time.time()
      self.pipe = QwenImage21Pipeline.from_pretrained(repo, text_encoder=None, dtype=DTYPE).to(DEVICE)
      self.repo = repo
      log.info("loaded denoiser in %.1fs", time.time() - t0)
    return self.pipe


def load_image(source):
  if source.startswith(("http://", "https://")):
    # NB: The ArtCraft CDN rejects urllib's default "Python-urllib" user agent with a 403.
    request = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
      data = response.read(MAX_REFERENCE_BYTES + 1)
  else:
    data = base64.b64decode(source)
  if len(data) > MAX_REFERENCE_BYTES:
    raise ValueError("reference image is too large")
  image = Image.open(io.BytesIO(data))
  image.load()
  return image


def encoder_images(enc, references):
  """Resize references exactly as `QwenImage21Pipeline.__call__` does before encoding them."""
  resized = []
  for img in references:
    if img.mode != "RGBA":
      img = img.convert("RGBA")
    width, height, _ = calculate_dimensions(OUTPUT_RESOLUTION * OUTPUT_RESOLUTION, img.size[0] / img.size[1])
    resized.append(enc.image_processor.resize(img, width=width, height=height))
  return resized


def free_memory():
  gc.collect()
  if DEVICE == "mps":
    torch.mps.empty_cache()


JOBS = {}
JOB_QUEUE = queue.Queue()
ENGINE = Engine()


def worker():
  while True:
    try:
      job = JOB_QUEUE.get(timeout=60)
    except queue.Empty:
      ENGINE.maybe_unload_idle()
      continue
    job.status = "running"
    req = job.request
    log.info("job %s: %dx%s x%d, %d refs, prompt=%r", job.id, req["width"] or 0, req["height"] or "auto",
             req["num_images"], len(req["images"]), req["prompt"][:80])
    try:
      ENGINE.generate(job)
      job.status = "completed"
    except Exception as err:  # noqa: BLE001 - report any failure back to the app
      log.exception("job %s failed", job.id)
      job.error = str(err) or type(err).__name__
      job.status = "failed"
      free_memory()


def parse_generate_request(body):
  model = body.get("model")
  if model not in MODELS:
    raise ValueError(f"unknown model {model!r}")
  prompt = str(body.get("prompt") or "").strip()
  if not prompt:
    raise ValueError("prompt is required")
  images = body.get("images") or []
  if not (isinstance(images, list) and all(isinstance(i, str) and i for i in images)):
    raise ValueError("images must be a list of URLs or base64 strings")
  if len(images) > MAX_REFERENCE_IMAGES:
    raise ValueError(f"at most {MAX_REFERENCE_IMAGES} reference images")
  width, height = body.get("width"), body.get("height")
  if images and width is None and height is None:
    pass  # follow the last reference image's aspect ratio
  else:
    width, height = int(width or 1024), int(height or 1024)
    if not (256 <= width <= MAX_SIDE and 256 <= height <= MAX_SIDE):
      raise ValueError(f"width/height must be 256..{MAX_SIDE}")
    width, height = width // 32 * 32, height // 32 * 32
  num_images = int(body.get("num_images", 1))
  if not 1 <= num_images <= MAX_IMAGES:
    raise ValueError(f"num_images must be 1..{MAX_IMAGES}")
  return {"model": model, "prompt": prompt, "width": width, "height": height, "num_images": num_images, "images": images}


class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/health":
      return self._json(200, {"ok": True, "device": DEVICE, "loaded": ENGINE.repo, "queued": JOB_QUEUE.qsize()})
    if self.path.startswith("/v1/jobs/"):
      job = JOBS.get(self.path.removeprefix("/v1/jobs/"))
      if job is None:
        return self._json(404, {"error": "job not found"})
      return self._json(200, {"status": job.status, "images": job.images if job.status == "completed" else [], "error": job.error})
    if self.path.startswith("/files/"):
      return self._file(self.path.removeprefix("/files/"))
    self._json(404, {"error": "not found"})

  def do_POST(self):
    if self.path != "/v1/generate":
      return self._json(404, {"error": "not found"})
    try:
      length = int(self.headers.get("Content-Length", 0))
      request = parse_generate_request(json.loads(self.rfile.read(length) or b"{}"))
    except (ValueError, TypeError, json.JSONDecodeError) as err:
      return self._json(400, {"error": str(err)})
    job = Job(request)
    JOBS[job.id] = job
    JOB_QUEUE.put(job)
    self._json(200, {"job_id": job.id})

  def _file(self, name):
    path = OUTPUT_DIR / name
    if not (FILE_NAME_RE.match(name) and path.is_file()):
      return self._json(404, {"error": "file not found"})
    data = path.read_bytes()
    self.send_response(200)
    self.send_header("Content-Type", "image/png")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def _json(self, status, payload):
    data = json.dumps(payload).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, fmt, *args):
    log.debug(fmt, *args)


def main():
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  try:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
  except OSError as err:
    if err.errno != errno.EADDRINUSE:
      raise
    log.error("port %d is already in use — the server is probably already running (check: lsof -i :%d)", PORT, PORT)
    raise SystemExit(1)
  threading.Thread(target=worker, daemon=True).start()
  log.info("serving on http://%s:%d (device=%s, output=%s)", HOST, PORT, DEVICE, OUTPUT_DIR)
  server.serve_forever()


if __name__ == "__main__":
  main()
