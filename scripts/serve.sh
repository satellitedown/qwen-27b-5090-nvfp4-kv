#!/usr/bin/env bash
# One GPU, one request. Does not modify or start any system service.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/serve.sh [additional SGLang arguments]
  PROFILE=nvfp4   Patched 262144-token profile (default)
  PROFILE=fp8     Unmodified 131072-token baseline
  MODEL_DIR      Local target checkpoint directory
  DRAFT_MODEL_DIR Local DFlash2 checkpoint directory
  PYTHON         Python executable (default: repository .venv/bin/python)
  CUDA_HOME      Full CUDA toolkit (default: repository .cuda-toolkit/nvidia/cu13)
  HOST / PORT    Bind address (default: 127.0.0.1 / 8000)
Stop with Ctrl-C. The API has no authentication; keep it on loopback.
HELP
  exit 0
fi

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/Huihui-Qwen3.8-27B-Abliterated-Gittensor-NVFP4}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-$ROOT/models/Qwen3.8-27B-DFlash2}"
export CUDA_HOME="${CUDA_HOME:-$ROOT/.cuda-toolkit/nvidia/cu13}"
export PATH="$(dirname -- "$PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_JOBS="${MAX_JOBS:-2}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment missing: run bash scripts/install.sh first." >&2
  exit 1
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" || ! -f "$CUDA_HOME/include/cuda.h" ]]; then
  echo "CUDA_HOME must contain bin/nvcc and include/cuda.h; run scripts/install.sh or set a full toolkit path." >&2
  exit 1
fi
for model in "$MODEL_DIR" "$DRAFT_MODEL_DIR"; do
  if [[ ! -f "$model/config.json" ]]; then
    echo "Missing model: $model. Run .venv/bin/python scripts/download_models.py or set model directory overrides." >&2
    exit 1
  fi
done

case "${PROFILE:-nvfp4}" in
  nvfp4)
    if [[ ! -f "$ROOT/runtime/sglang/kernels/ops/kvcache/nvfp4_dequant.py" ]]; then
      echo "Patched runtime missing: run .venv/bin/python scripts/apply_patch.py." >&2
      exit 1
    fi
    export PYTHONPATH="$ROOT/runtime"
    export SGLANG_FLASHINFER_WORKSPACE_SIZE=134217728
    profile_args=(
      --served-model-name qwen3.8-27b-abliterated-256k
      --context-length 262144
      --kv-cache-dtype nvfp4
      --prefill-attention-backend flashinfer
      --decode-attention-backend trtllm_mha
      --speculative-attention-mode decode
      --page-size 64
      --max-mamba-cache-size 4
      --mem-fraction-static 0.957
      --max-total-tokens 263168
      --speculative-draft-kv-cache-dtype fp8_e4m3
    )
    ;;
  fp8)
    unset PYTHONPATH SGLANG_FLASHINFER_WORKSPACE_SIZE
    profile_args=(
      --served-model-name qwen3.8-27b-abliterated
      --context-length 131072
      --kv-cache-dtype fp8_e4m3
      --max-mamba-cache-size 8
      --mem-fraction-static 0.90
    )
    ;;
  *)
    echo "PROFILE must be nvfp4 or fp8." >&2
    exit 1
    ;;
esac

exec "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --attention-backend flashinfer \
  --chunked-prefill-size 1024 \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype bfloat16 \
  --max-running-requests 1 \
  --language-only \
  --mm-feature-transport cpu \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL_DIR" \
  --speculative-dflash-block-size 8 \
  --speculative-draft-window-size 8192 \
  --speculative-draft-model-quantization fp8 \
  --speculative-draft-attention-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"reasoning_effort":"medium","preserve_thinking":true}' \
  --enable-metrics \
  "${profile_args[@]}" \
  "$@"
