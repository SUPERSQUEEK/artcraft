"""On-device image server for ArtCraft's "Local" provider (Qwen-Image 2.1 Turbo on Apple Silicon).

ArtCraft enqueues a job, polls it, then downloads the finished PNGs over HTTP:

    POST /v1/generate   {"model", "prompt", "width", "height", "num_images"} -> {"job_id"}
    GET  /v1/jobs/<id>  -> {"status": queued|running|completed|failed, "images": [urls], "error"}
    GET  /files/<name>  -> image/png
    GET  /health        -> {"ok": true, ...}

Jobs run one at a time. The BF16 weights (17.5 GB text encoder + 14.2 GB transformer)
don't fit in 36 GB unified memory together, so the transformer + VAE stay resident and
the text encoder is loaded per job, used, and freed before denoising.
"""

import gc
import json
import logging
import os
import queue
import random
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

HOST = os.environ.get("LOCAL_IMAGE_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCAL_IMAGE_SERVER_PORT", "8189"))
OUTPUT_DIR = Path(os.environ.get("LOCAL_IMAGE_SERVER_OUTPUT", Path.home() / "Pictures" / "ArtCraft Local"))
IDLE_UNLOAD_SECS = int(os.environ.get("LOCAL_IMAGE_SERVER_IDLE_UNLOAD_SECS", "900"))

MODELS = {"qwen_image_2p1_turbo": "Qwen/Qwen-Image-2.1-Turbo"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_SIDE = 2048
MAX_IMAGES = 4
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
    t0 = time.time()
    embeds, mask = self._encode(repo, req["prompt"])
    log.info("encoded prompt in %.1fs", time.time() - t0)

    pipe = self._denoiser(repo)
    base_seed = random.randrange(2**31)
    for i in range(req["num_images"]):
      t1 = time.time()
      image = pipe(
        prompt_embeds=embeds,
        prompt_embeds_mask=mask,
        width=req["width"],
        height=req["height"],
        use_kv_cache=True,
        generator=torch.Generator("cpu").manual_seed(base_seed + i),
      ).images[0]
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

  def _encode(self, repo, prompt):
    enc = QwenImage21Pipeline.from_pretrained(repo, transformer=None, vae=None, dtype=DTYPE).to(DEVICE)
    try:
      with torch.no_grad():
        embeds, mask, _ = enc.encode_prompt(prompt, device=DEVICE)
      embeds = embeds.detach().clone()
      mask = mask.detach().clone() if mask is not None else None
    finally:
      del enc
      free_memory()
    return embeds, mask

  def _denoiser(self, repo):
    if self.pipe is None or self.repo != repo:
      self.pipe = None
      free_memory()
      t0 = time.time()
      self.pipe = QwenImage21Pipeline.from_pretrained(repo, text_encoder=None, dtype=DTYPE).to(DEVICE)
      self.repo = repo
      log.info("loaded denoiser in %.1fs", time.time() - t0)
    return self.pipe


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
    log.info("job %s: %r", job.id, job.request)
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
  width, height = int(body.get("width", 1024)), int(body.get("height", 1024))
  if not (256 <= width <= MAX_SIDE and 256 <= height <= MAX_SIDE):
    raise ValueError(f"width/height must be 256..{MAX_SIDE}")
  num_images = int(body.get("num_images", 1))
  if not 1 <= num_images <= MAX_IMAGES:
    raise ValueError(f"num_images must be 1..{MAX_IMAGES}")
  return {"model": model, "prompt": prompt, "width": width // 16 * 16, "height": height // 16 * 16, "num_images": num_images}


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
  threading.Thread(target=worker, daemon=True).start()
  log.info("serving on http://%s:%d (device=%s, output=%s)", HOST, PORT, DEVICE, OUTPUT_DIR)
  ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
  main()
