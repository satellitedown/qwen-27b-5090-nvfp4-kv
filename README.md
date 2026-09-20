# NVFP4 weights + KV, DFlash2, patched SGLang

A custom **SGLang runtime patch** for Huihui Qwen3.8-27B with NVFP4 target weights, NVFP4 KV cache, and DFlash2 speculative decoding. Tested on **one RTX 5090 with 32 GB VRAM**, including a **260,000-token prompt**. This is a deployment recipe, not a new model.

## Why was the patch needed?

Quantized weights alone do not solve long-context memory use. The pinned **SGLang 0.5.20** runtime had two blockers:

1. **Native NVFP4 attention rejected speculative verification.** DFlash2 needs the target to verify a block of tokens, not just decode one. The patch adds causal masks and multi-token routing in `trtllm_mha_backend.py`, reusing FlashInfer's native NVFP4 kernel with CUDA graphs. Support is scoped to fixed-length, causal, top-1 verification.
2. **Prefill exhausted VRAM despite the compressed cache fitting.** The old path gathered and dequantized the cached prefix into large temporary tensors. A new Triton kernel, `nvfp4_dequant.py`, gathers and converts directly into the shared FP8 prefill workspace, wired through the cache quantization and memory-pool code.

**The benefit: compressed target KV and DFlash2 can work together at long context without those temporary allocations exhausting memory.** Launch flags alone were not enough. Prefill still uses a converted workspace; this is not an entirely 4-bit computation pipeline.

**[Read the SGLang patch](patches/sglang-nvfp4-kv.patch)** · [Exact runtime/model pins](runtime-manifest.json) · [Launch settings](scripts/serve.sh)

## Will it work on other hardware?

**Only RTX 5090 32 GB on Linux x86_64 is verified.** Other NVIDIA GPUs—including other Blackwell cards—are untested. Native NVFP4 kernel support, backend routing, and memory settings need verification; more VRAM alone does not guarantee compatibility. This CUDA-specific recipe is not a drop-in for AMD, Apple GPUs, or CPUs.

The installer targets Python 3.12 and a pinned SGLang wheel. Other SGLang versions require review and testing, not blindly applying this patch.

## Run

Requires a CUDA 13-compatible NVIDIA driver, [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, GNU `patch`, and a C/C++ toolchain. The installer supplies the Python environment and CUDA development packages; model downloads use pinned revisions.

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
