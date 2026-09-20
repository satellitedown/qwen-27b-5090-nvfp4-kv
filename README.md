# NVFP4 weights + KV, DFlash2, patched SGLang

I could only reach **around 160K context** with Huihui Qwen3.8-27B before my **32 GB RTX 5090 ran out of VRAM**. The goal was to fit the full context window without giving up DFlash2 speculative decoding.

**NVFP4 target weights + NVFP4 KV + a small SGLang patch** made the full **262,144-token window** possible on the same card. Tested with a **260,000-token prompt**, leaving room for output. This is a runtime patch and deployment recipe, not a new model.

## Why was the patch needed?

Quantized weights alone do not solve long-context memory use. The pinned **SGLang 0.5.20** runtime had two blockers:

1. **Native NVFP4 attention rejected speculative verification.** DFlash2 needs the target to verify a block of tokens, not just decode one. The patch adds causal masks and multi-token routing in `trtllm_mha_backend.py`, reusing FlashInfer's native NVFP4 kernel with CUDA graphs. Support is scoped to fixed-length, causal, top-1 verification.
2. **Prefill exhausted VRAM despite the compressed cache fitting.** The old path gathered and dequantized the cached prefix into large temporary tensors. A new Triton kernel, `nvfp4_dequant.py`, gathers and converts directly into the shared FP8 prefill workspace, wired through the cache quantization and memory-pool code.

**[Read the SGLang patch](patches/sglang-nvfp4-kv.patch)** · [Exact runtime/model pins](runtime-manifest.json) · [Launch settings](scripts/serve.sh)

## Will it work on other hardware?

**NVIDIA-only, built and tested for the RTX 5090 32 GB.** Other NVIDIA GPUs are untested. **No Apple, AMD, or CPU support.**

## Run

Requires Linux x86_64, a CUDA 13-compatible NVIDIA driver, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, GNU `patch`, and a C/C++ toolchain. The installer supplies the pinned Python 3.12 environment and CUDA development packages; model downloads use pinned revisions.

```bash
git clone https://github.com/satellitedown/qwen-27b-5090-nvfp4-kv.git
cd qwen-27b-5090-nvfp4-kv
bash scripts/install.sh
.venv/bin/python scripts/download_models.py
bash scripts/serve.sh
```

First startup can take several minutes to compile kernels. The patch uses an isolated runtime copy, leaving global installations untouched. OpenAI-compatible endpoint: **`http://127.0.0.1:8000/v1`**, model **`qwen3.8-27b-abliterated-256k`**. No authentication: keep it on loopback. Stop with Ctrl-C.

## Results and limits

- **~239 generated tokens/s after 260K input**, with **~134 s cold time to first token**, on a synthetic coding task: two-run mean, thinking disabled. Decode speed excludes prompt processing. [Measurements](results/nvfp4.json).
- Numerical checks, CUDA graph replay, and six-marker recall at 260K passed. [Kernel checks](results/kernel-check.json) · [Packaging verification](results/publication-smoke.json).
- **Experimental:** tight VRAM headroom and errors in a small coding check—not a broad quality guarantee. One request at a time, text-only. The configured **262,144-token budget includes output**; check actual cache allocation at startup because available memory can reduce capacity.

## Credits

Model/runtime credit: Qwen, Huihui AI, heswithme, Gittensor, Inco/DFlash, SGLang, FlashInfer, PyTorch, and NVIDIA. This repo contributes the patch and recipe; no model weights are distributed. [Apache-2.0](LICENSE) · [Attribution](NOTICE).
