#!/usr/bin/env bash
# Start the on-device image server used by ArtCraft's "Local" provider.
# First run creates a venv and installs dependencies (needs `uv`).
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv="${LOCAL_IMAGE_SERVER_VENV:-${here}/.venv}"

if [[ ! -x "${venv}/bin/python" ]]; then
  uv venv -p 3.12 "${venv}"
  uv pip install -p "${venv}" -r "${here}/requirements.txt"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "${venv}/bin/python" "${here}/server.py"
