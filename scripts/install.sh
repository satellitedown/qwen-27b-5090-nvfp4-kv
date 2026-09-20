#!/usr/bin/env bash
# Install the captured Linux x86_64 / Python 3.12 environment, not a fresh solve.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This recipe is pinned for Linux x86_64." >&2
  exit 1
fi
for command in uv patch c++; do
  if ! command -v "$command" >/dev/null; then
    echo "Required command not found: $command (see README prerequisites)." >&2
    exit 1
  fi
done
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
uv pip install --python "$ROOT/.venv/bin/python" --no-deps -r "$ROOT/requirements.lock"
uv pip install --python "$ROOT/.venv/bin/python" --target "$ROOT/.cuda-toolkit" \
  --no-deps -r "$ROOT/requirements-cuda.lock"
# NVIDIA's wheels omit the toolkit/linker aliases expected by CUDA JIT builds.
ln -sfn lib "$ROOT/.cuda-toolkit/nvidia/cu13/lib64"
ln -sfn libcudart.so.13 "$ROOT/.cuda-toolkit/nvidia/cu13/lib/libcudart.so"
"$ROOT/.venv/bin/python" "$ROOT/scripts/apply_patch.py"
printf '\nInstalled. Next: .venv/bin/python scripts/download_models.py\nThen: bash scripts/serve.sh\n'
