# Local image server

Runs Qwen-Image 2.1 Turbo on this Mac for ArtCraft's **Local** provider
(model picker → "Qwen-Image 2.1 Turbo", provider "This Mac (local)").

```
./local_image_server/serve.sh
```

The first run creates `.venv` (needs `uv`) and the first job downloads ~32 GB of
weights into the Hugging Face cache. Keep the server running while you generate.

| Setting                               | Default                    |
|---------------------------------------|----------------------------|
| `LOCAL_IMAGE_SERVER_PORT`             | `8189`                     |
| `LOCAL_IMAGE_SERVER_OUTPUT`           | `~/Pictures/ArtCraft Local`|
| `LOCAL_IMAGE_SERVER_IDLE_UNLOAD_SECS` | `900`                      |
| `ARTCRAFT_LOCAL_IMAGE_SERVER_URL` (app side) | `http://127.0.0.1:8189` |

## How it fits together

1. The app's `handle_local` provider posts the prompt to `/v1/generate` and stores a
   task with provider `local`.
2. The third-party polling thread polls `/v1/jobs/<id>`.
3. On completion it downloads the PNGs from `/files/...`, uploads them to the
   ArtCraft library, and notifies the frontend — same end state as a cloud job.

Memory: the BF16 text encoder (17.5 GB) and transformer (14.2 GB) don't fit in 36 GB
together, so the transformer stays loaded and the text encoder is loaded per job.
