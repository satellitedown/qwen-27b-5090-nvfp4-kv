#!/usr/bin/env bash
# Interactive front end for the existing, pinned installation and model scripts.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT/.tools/bin:$PATH"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: bash setup.sh\n\nChoose 1 to install the runtime and download both models; choose 3 to start.\nRequires Linux x86_64 and a working NVIDIA driver. Drivers are never changed.\n'
  exit 0
fi
if [[ $# -ne 0 || ! -t 0 ]]; then
  echo "Run bash setup.sh in an interactive terminal (or use --help)." >&2
  exit 2
fi
active_step=""
interrupt_setup() {
  trap '' INT
  if [[ -n "$active_step" ]]; then
    kill -TERM -- "-$active_step" 2>/dev/null || true
    wait "$active_step" 2>/dev/null || true
  fi
  printf '\nInterrupted. Rerun setup and choose 1 or 2 to resume model downloads.\n'
  exit 130
}
trap interrupt_setup INT

run_step() {
  # A separate process group lets Ctrl-C stop downloader workers immediately.
  # Waiting in Bash keeps its interrupt trap responsive during long downloads.
  setsid "$@" &
  active_step=$!
  local status=0
  wait "$active_step" || status=$?
  active_step=""
  return "$status"
}

check_hardware() {
  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This recipe requires Linux x86_64; Apple, AMD and CPU inference are not supported." >&2
    return 1
  fi
  if ! command -v nvidia-smi >/dev/null; then
    echo "NVIDIA driver tools are missing. Install your NVIDIA driver, then rerun setup." >&2
    return 1
  fi
  local gpus
  if ! gpus="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"; then
    echo "The NVIDIA driver is not working. Fix it before continuing; this menu does not install drivers." >&2
    return 1
  fi
  printf '\nDetected GPU(s):\n%s\n' "$gpus"
  if [[ "$gpus" != *"RTX 5090"* ]]; then
    echo "Warning: this preset is tested only on the RTX 5090 32 GB. Other NVIDIA GPUs are untested."
  fi
}

ensure_system_tools() {
  local command answer
  local missing=() package_command=() privilege=()
  for command in curl patch c++ setsid; do
    command -v "$command" >/dev/null || missing+=("$command")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi
  printf '\nMissing system tools: %s\n' "${missing[*]}"
  if command -v apt-get >/dev/null; then
    package_command=(apt-get install -y build-essential patch curl util-linux)
  elif command -v dnf >/dev/null; then
    package_command=(dnf install -y gcc gcc-c++ make patch curl util-linux)
  elif command -v pacman >/dev/null; then
    package_command=(pacman -S --needed base-devel patch curl util-linux)
  else
    echo "Install curl, GNU patch, a C/C++ build toolchain, and util-linux (setsid), then choose 1 again." >&2
    return 1
  fi
  if [[ $EUID -ne 0 ]]; then
    if ! command -v sudo >/dev/null; then
      echo "Ask your administrator to install the missing tools, then choose 1 again." >&2
      return 1
    fi
    privilege=(sudo)
  fi
  printf 'System package command: %s\n' "${privilege[*]} ${package_command[*]}"
  echo "On apt-based systems, package lists will also be refreshed first."
  if ! read -r -p "Install these system build tools? [y/N] " answer; then
    return 1
  fi
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "No system packages changed."; return 1 ;;
  esac
  if [[ "${package_command[0]}" == apt-get ]]; then
    "${privilege[@]}" apt-get update || return 1
  fi
  "${privilege[@]}" "${package_command[@]}" || return 1
  for command in curl patch c++ setsid; do
    if ! command -v "$command" >/dev/null; then
      echo "Still missing $command; install it and choose 1 again." >&2
      return 1
    fi
  done
}

ensure_uv() {
  if command -v uv >/dev/null; then
    return 0
  fi
  local installer status
  echo "Installing uv from https://astral.sh/uv/install.sh into .tools/bin (no shell profile changes)."
  installer="$(mktemp)" || return 1
  if ! curl --fail --show-error --location --proto '=https' --tlsv1.2 \
    https://astral.sh/uv/install.sh --output "$installer"; then
    rm -f -- "$installer"
    return 1
  fi
  status=0
  UV_UNMANAGED_INSTALL="$ROOT/.tools/bin" sh "$installer" || status=$?
  rm -f -- "$installer"
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
  command -v uv >/dev/null
}

download_models() {
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "The Python environment is not installed yet. Choose 1 first." >&2
    return 1
  fi
  run_step "$ROOT/.venv/bin/python" "$ROOT/scripts/download_models.py"
}

install_everything() {
  check_hardware || return 1
  ensure_system_tools || return 1
  ensure_uv || return 1
  printf '\n[1/2] Installing the pinned runtime, CUDA toolkit and SGLang patch...\n'
  run_step bash "$ROOT/scripts/install.sh" || return 1
  printf '\n[2/2] Downloading the target model and DFlash2 from Hugging Face...\n'
  download_models || return 1
  printf '\nSetup complete. Choose 3 to start the server.\n'
}

while true; do
  printf '\nQwen 27B / RTX 5090 setup\n'
  printf 'NVFP4 weights + KV, DFlash2, patched SGLang\n\n'
  printf '  1) Set up everything (runtime + both model downloads)\n'
  printf '  2) Download / resume models only\n'
  printf '  3) Start the server (Ctrl-C to stop)\n'
  printf '  0) Exit\n\n'
  printf 'First setup downloads about 22 GB of model weights, plus runtime packages.\n'
  printf 'Keep enough disk space free. Model files stay in models/ inside this repository.\n'
  if ! read -r -p "Choose an option: " choice; then
    printf '\n'
    exit 0
  fi
  case "$choice" in
    1)
      if ! install_everything; then
        printf '\nSetup did not finish. Fix the error above, then choose 1 again.\n' >&2
      fi
      ;;
    2)
      if ! download_models; then
        printf '\nDownload did not finish. Fix the error above, then retry.\n' >&2
      fi
      ;;
    3)
      if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
        echo "Choose 1 to install everything first." >&2
      elif check_hardware; then
        echo "Starting the NVFP4 server. First-start CUDA compilation can take several minutes."
        exec env PROFILE=nvfp4 bash "$ROOT/scripts/serve.sh"
      fi
      ;;
    0|q|Q) exit 0 ;;
    *) echo "Choose 1, 2, 3, or 0." ;;
  esac
done
