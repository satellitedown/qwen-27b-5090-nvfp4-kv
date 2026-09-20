# Qwen 27B + DFlash2: 262,144-token context on one RTX 5090

**An experimental SGLang recipe with NVFP4 target KV, demonstrated with 260,000 actual prompt tokens on a 32 GB RTX 5090.** The measured coding sample decoded at **239.20 tokens/s** at that depth—but cold time to first token was **133.71 seconds**, and a small coding check found errors absent from the FP8 baseline.

This repository contributes a pinned serving recipe, a small SGLang runtime patch, a reproducible benchmark harness, and saved local measurements. It does **not** provide a new model, quantization, or DFlash algorithm. The target is the existing **Huihui abliterated Qwen3.8-27B checkpoint packaged by heswithme**, not the unmodified Qwen model. Model authors and runtime projects are credited below.

The experimental profile is the launcher's default to make this recipe easy to reproduce. **Use `PROFILE=fp8` for the more conservative, unpatched 131,072-token baseline.** Neither the throughput nor sampled recall establishes general speed or quality equivalence.

## What runs

| Setting | Experimental `nvfp4` | Baseline `fp8` |
|---|---|---|
| Total context: prompt **plus** output | 262,144 tokens | 131,072 tokens |
| Target weights | Published ModelOpt NVFP4 checkpoint | Same checkpoint |
| Target KV cache | NVFP4, including scale overhead | FP8 E4M3 |
| DFlash2 weights / KV | FP8 at load / FP8 E4M3 | Same |
| Speculation | `DFLASH`, block size 8, draft window 8,192 | Same |
| Attention | FlashInfer prefill; `trtllm_mha` decode/verification | FlashInfer |
| Recurrent state | BF16, `extra_buffer_lazy`, 4 slots | BF16, `extra_buffer_lazy`, 8 slots |
| Static memory fraction | 0.957 | 0.90 |
| Token pool | Explicit cap 263,168 | Automatically sized |
| Prefill / graphs | Chunk 1,024; prefill graphs off; decode graph batch size 1 | Same |
| Concurrency / modality | One running request; language-only | Same |
| Runtime | Patched copy under `runtime/sglang` | Installed SGLang, without the patch |

The NVFP4 profile also uses page size 64 and a **128 MiB FlashInfer attention workspace**. This attention workspace is distinct from the FP8 KV workspace used during prefill.

Three different precision choices matter:

- **Target weights:** already quantized by the checkpoint publisher; this repository does not requantize them. The checkpoint has a quantized LM head and no native MTP head; not every tensor is NVFP4.
- **Draft weights:** the DFlash2 download is BF16 and is quantized to FP8 when loaded. DFlash block size 8 means **seven draft tokens plus one verification token**, not eight proposed tokens. The drafter's 8,192-token window is not the target's context limit.
- **Target KV:** this is the experimental FP8-to-NVFP4 change. Draft KV stays FP8, and recurrent state stays BF16. The attention backend also changes, so this is a comparison of complete serving profiles, not an isolated precision experiment.

### Pinned models

| Role | Hugging Face model | Revision |
|---|---|---|
| Target | [`heswithme/Huihui-Qwen3.8-27B-Abliterated-Gittensor-Style-NVFP4-RTX5090`](https://huggingface.co/heswithme/Huihui-Qwen3.8-27B-Abliterated-Gittensor-Style-NVFP4-RTX5090/tree/41c5fdfad0cbe0c784981b3e44d49c164a9bd893) | `41c5fdfad0cbe0c784981b3e44d49c164a9bd893` |
| Draft | [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/015e795645c74b1a0eeef3b570031fb62e769bc5) | `015e795645c74b1a0eeef3b570031fb62e769bc5` |

The downloader uses these revisions, not moving branch tips. Its default destinations are `models/Huihui-Qwen3.8-27B-Abliterated-Gittensor-NVFP4` and `models/Qwen3.8-27B-DFlash2`. Weights are downloaded separately and are not committed here.

## Prerequisites

- **Linux x86_64, Python 3.12, and one RTX 5090 with 32 GB VRAM.** This recipe was measured on that hardware, not qualified across other GPUs, Windows, or macOS.
- A working NVIDIA driver. The measured machine used **610.57.04**; this is an observed version, not a claimed minimum.
- A **CUDA 13 toolkit with `nvcc`, headers, and development libraries**, not merely a driver. The installer supplies pinned compiler/header packages under `.cuda-toolkit/nvidia/cu13`, separate from PyTorch's CUDA runtime. The launcher discovers this toolkit; `CUDA_HOME` can override it. A working `nvidia-smi`, or the CUDA version displayed by it, does **not** prove that a toolkit is installed.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, Git, GNU `patch`, and a C/C++ build toolchain (`c++` on `PATH`). Allow disk and system RAM for the packages, both model downloads, and CUDA JIT compilation. No minimum system-RAM requirement has been established. The measured machine had approximately 60 GiB of system RAM.
- Enough free VRAM for a tightly packed, single-request server. Desktop applications share the GPU; avoid other GPU workloads during startup and long-context inference.

Runtime anchors: **SGLang 0.5.20**, **PyTorch 2.13.0+cu130**, **FlashInfer 0.6.18**, and **Transformers 5.12.1**. [`requirements.lock`](requirements.lock) captures the full installed environment; installation uses `--no-deps` intentionally to restore these exact pins rather than resolve a new dependency set. [`requirements-cuda.lock`](requirements-cuda.lock) pins the separate compiler/header toolkit, and [`runtime-manifest.json`](runtime-manifest.json) records provenance. The SGLang base is identified by its **PyPI wheel and source hashes**, not an unverified Git commit.

## Install and start

Run from the repository root. The scripts create a local `.venv` and an ignored, patched runtime copy; they do not require editing a global SGLang installation.

```bash
git clone https://github.com/satellitedown/qwen-27b-5090-nvfp4-kv.git
cd qwen-27b-5090-nvfp4-kv

bash scripts/install.sh
.venv/bin/python scripts/download_models.py

# Foreground server; first-start CUDA compilation can take several minutes.
bash scripts/serve.sh
```

`install.sh` installs the pinned environment and CUDA development packages, creates the linker aliases required by CUDA JIT builds, then applies the hash-checked patch to `runtime/sglang`. The packaged scripts were verified with a fresh project-local environment and toolkit on the original machine, reusing existing SHA-256-verified model downloads and package/JIT caches. Both profiles served the OpenAI API; the NVFP4 profile also recovered all six markers from a 260,000-token prompt. This is not a clean-machine qualification or a rerun of the full comparison below.

The server defaults to **`127.0.0.1:8000` without authentication**. Keep it on loopback. Changing `HOST` does not add authentication; use appropriate access controls before exposing the API.

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-abliterated-256k",
    "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
    "temperature": 0,
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Switch to the FP8 baseline

**Run one server at a time.** Stop the foreground server with Ctrl-C, wait for its process to exit and release VRAM, then start:

```bash
PROFILE=fp8 bash scripts/serve.sh
```

Use model ID `qwen3.8-27b-abliterated` in baseline API requests. Selecting another model ID in a client does not switch the running profile. Return to the experiment by stopping the baseline and running `PROFILE=nvfp4 bash scripts/serve.sh`.

Launcher environment overrides are `PROFILE`, `PYTHON` (default `.venv/bin/python`), `MODEL_DIR`, `DRAFT_MODEL_DIR`, `HOST`, `PORT`, and `CUDA_HOME`. Use the pinned model snapshots even when pointing at existing downloads. The NVFP4 launcher selects the patched runtime through `PYTHONPATH`; the baseline does not.

**Context accounting:** 262,144 is the total sequence budget, not an input allowance. A 260,000-token prompt leaves 2,144 tokens for output. Reserving 8,192 output tokens leaves at most **253,952 prompt tokens**, including chat templates and tool definitions. The 263,168-token physical cache pool is a separate allocation, not an increased context limit.

## Measured results

These are **two-run arithmetic means of single-request coding decode throughput**, with thinking disabled. Actual prompt token counts were checked against server usage. Shared lengths used identical tokenized inputs; generated outputs and completion lengths could differ.

| Actual prompt tokens | FP8 target KV, tok/s | NVFP4 target KV, tok/s |
|---:|---:|---:|
| 8,192 | 376.94 | 302.68 |
| 32,768 | 309.36 | 340.67 |
| 122,000 | 285.36 | 292.05 |
| 196,608 | — | 291.85 |
| 260,000 | — | 239.20 |

A dash means the prompt exceeds the baseline's configured context window, not a measured zero or a universal FP8 hardware limit. The measured baseline allocated 144,681 cache tokens; allocation is distinct from its 131,072-token admission limit.

**At 260,000 input tokens, coding time to first token was 133.708633 s cold and 0.100108 s with the prefix cached.** Decode throughput excludes this initial wait. “Cold” here means a flushed prefix cache after kernel warmup—not a cold process, model load, or first CUDA compilation.

### Quality and memory caveats

- **Recall:** the NVFP4 profile recovered all six markers at all five depths, cold and warm: **60/60 values**, including both 260,000-token runs. This is sparse synthetic retrieval, not proof of general long-context comprehension or precision equivalence.
- **Matched coding checks:** FP8 passed **30/30**, NVFP4 **27/30**: five behavioral checks on each of six outputs across the three shared depths. Three NVFP4 outputs mishandled expired-most-recently-used cleanup; all six FP8 outputs passed. Across all five NVFP4 depths the count was **44/50**. This is one LRU/TTL coding task, not a broad quality benchmark, and it cannot isolate KV precision from backend effects.
- **VRAM:** minimum sampled free memory was **616 MiB**, at two-second intervals. This does not capture every instantaneous peak or guarantee equivalent headroom on another desktop.
- **Scope:** one GPU, one running request, language-only, synthetic operational-record filler, and a specific abliterated checkpoint. No broad concurrency, multimodal, safety, or quality claims. Abliteration reduces refusal behavior; treat model outputs and generated code as untrusted.

Traceable saved records: [`results/summary.json`](results/summary.json), [`results/fp8.json`](results/fp8.json), [`results/nvfp4.json`](results/nvfp4.json), [`results/quality.json`](results/quality.json), and [`results/kernel-check.json`](results/kernel-check.json). Public artifacts omit private deployment details; they describe historical measurements, not a live dashboard.

[`results/publication-smoke.json`](results/publication-smoke.json) records the separate packaging check: numerical kernel verification, OpenAI API requests, 8K coding/recall on both profiles, and 260K recall on NVFP4. On that run, runtime memory profiling reduced the allocated pool to **261,312 tokens** despite the requested cap of 263,168. The 260K request passed with a 192-token output allowance; that run does **not** independently establish the full 262,144-token window. Check your actual allocation at startup.

## Reproduce the benchmark

Start one matching server as above. Run the client in a second terminal from the repository root. Use **fresh output filenames**: the harness refuses to overwrite an existing result. These requests flush the server cache, so do not benchmark a server serving other work.

```bash
mkdir -p runs

# With PROFILE=fp8 running:
.venv/bin/python scripts/benchmark.py \
  --arm fp8 --url http://127.0.0.1:8000 \
  --depths 8192 32768 122000 --cases code recall \
  --repeats 2 --max-tokens 1536 --out runs/fp8.json

# Stop FP8; start PROFILE=nvfp4 before running this command:
.venv/bin/python scripts/benchmark.py \
  --arm nvfp4 --url http://127.0.0.1:8000 \
  --depths 8192 32768 122000 196608 260000 --cases code recall \
  --repeats 2 --max-tokens 1536 --out runs/nvfp4.json
```

`--arm` labels results; it does not reconfigure the server. `--model` can override the local target tokenizer directory; its default matches the downloader. `--url` is the server root, **without `/v1`**, because the harness uses SGLang's native endpoints.

The protocol is deliberately narrow:

1. Build deterministic synthetic operational records and tokenize the pinned target's chat template with **`enable_thinking=False`**. Fill to the exact requested input length. Send token IDs directly to `/generate` and reject a usage-count mismatch.
2. Warm kernels with a separate short request. Flush the prefix cache before each case/depth, then run two repetitions: first cold-cache, second warm-cache. Concurrency is one and temperature is zero.
3. For coding, ask for a bounded Python LRU cache with TTL, an injected clock, inclusive expiry, overwrite/LRU behavior, expired-entry cleanup, and deterministic unit tests. Cap generation at **1,536 tokens**. For recall, place six depth-specific secret markers near 1%, 10%, 25%, 50%, 75%, and 95% of the archive and cap output at **192 tokens**.
4. Measure TTFT from the client request start to the first streamed event with an increased completion count. Compute decode rate as **`(last completion count − first completion count) / (last token-event time − first token-event time)`**, then average the two per-run rates. Speculative events can contain multiple tokens; this is not a first-token-subtracted wall-clock average. TTFT includes HTTP overhead and prompt processing.
5. Save server metadata, input token hashes, generated text, token counts, finish metadata, timing, syntax checks, and recall scores. The harness **does not execute generated code or automatically reproduce the separate behavioral quality analysis**; the original five-check results are preserved in `results/quality.json`.

The launcher's normal reasoning settings are not the benchmark settings: **all generations in this comparison had thinking disabled**. Rates can change substantially with output content, reasoning mode, acceptance rate, runtime, GPU contention, and cache state.

## What the runtime patch changes

[`patches/sglang-nvfp4-kv.patch`](patches/sglang-nvfp4-kv.patch) is deliberately scoped to the pinned SGLang base:

1. Enable **native causal, fixed-length, top-1 NVFP4 multi-token verification** through the `trtllm_mha` backend, retaining CUDA graph replay rather than expanding the entire target cache for every decode step.
2. Fuse **KV gather and NVFP4 dequantization directly into the existing FP8 prefill workspace**. This removes the full-prefix gather/BF16/FP8 temporary tensors that exhausted VRAM at long input lengths. Prefill still uses an FP8 representation; this is not an all-native-NVFP4 prefill kernel.

The copied runtime is selected only for the experimental profile. Hash checks guard against applying this patch to another source version. Do not bypass a mismatch or install a newer SGLang release underneath it and assume compatibility.

### Numerical verification

Stop the server first to free VRAM. Select the installed toolkit explicitly for this standalone probe:

```bash
export CUDA_HOME="$PWD/.cuda-toolkit/nvidia/cu13"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1 PYTHONPATH="$PWD/runtime" \
  .venv/bin/python scripts/verify_kernels.py
```

The saved probe compared native causal attention with a dequantized reference at query lengths **1, 4, 6, and 8**, including CUDA graph replay. Fused gather/conversion was bit-exact against the finite BF16-to-FP8 reference, including signed zero and saturation, at **0, 1, 67, 4,097, and 260,003 gathered tokens**. Warmed conversion allocated **zero additional PyTorch CUDA tensor bytes**. These are numerical/allocator checks of the patched path, not end-to-end model quality equivalence or a claim of zero total CUDA memory use. See [`results/kernel-check.json`](results/kernel-check.json).

## Troubleshooting and operating limits

- **`nvcc` missing, CUDA headers missing, or JIT failures:** check that `CUDA_HOME` names a complete compatible toolkit and that `$CUDA_HOME/bin` is on `PATH`. Driver installation alone is insufficient.
- **Host RAM pressure during startup:** keep `MAX_JOBS=2` and `FLASHINFER_NVCC_THREADS=1` (the launcher sets these). Unrestricted compilation caused system-memory pressure in the original deployment. A separate user-systemd cgroup can protect the terminal; the original used `MemoryMax=40G` and `MemorySwapMax=2G`. Those are deployment-specific limits, **not minimum RAM requirements**, and a restrictive limit can terminate compilation. The portable launcher runs directly rather than requiring systemd.
- **GPU out of memory or insufficient physical cache capacity:** stop other model servers and GPU applications first. Check the startup log's **allocated `#tokens`**, not just `--context-length` or `--max-total-tokens`: SGLang can clamp the requested 263,168-token pool to a smaller profiled allocation when desktop VRAM use changes. A pool below 262,144 does not establish full-window capacity. Prefer the FP8 baseline or reduce the experimental context **and** token-pool allocation; reducing only the context flag need not shrink an explicitly sized pool. Lowering the static memory fraction can leave too few cache tokens for 262,144 context. Any changed configuration needs its own capacity and long-prompt verification.
- **Do not “save memory” by shrinking the wrong buffers:** three recurrent-state slots were rejected for this cache strategy; four are required for this single-request profile. A 64 MiB attention workspace passed short requests but failed at 122,000-token prefill. Keep the 128 MiB attention workspace. Before fused conversion, a large pool could fit at startup and still OOM during 196,608-token prefill.
- **Slow first request:** model loading and CUDA compilation are separate from the reported cold-cache TTFT. Even after warmup, a new 260K prefix took over two minutes before the first coding token. Prefix reuse is much faster but is not cold ingestion performance.
- **Unexpected code or retrieval results:** retain the output and exact runtime/profile. The FP8 baseline is a comparison point, not a guarantee of correctness. Do not execute untrusted generated programs outside a sandbox.

## Attribution and licensing

This is an independent recipe, not an official Qwen, Huihui, Gittensor, heswithme, Inco, NVIDIA, or SGLang release.

- **Qwen team:** base model architecture and weights; **[Huihui AI](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated):** abliterated source.
- **[heswithme](https://huggingface.co/heswithme/Huihui-Qwen3.8-27B-Abliterated-Gittensor-Style-NVFP4-RTX5090):** independent NVFP4 quantization, validation, and packaging. **[Gittensor Model Hub](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090):** quantization recipe/layout and the target's published LM-head/tokenizer provenance. See the target model card, `PROVENANCE.json`, `SHA256SUMS`, and `LICENSE` in the downloaded snapshot.
- **[Inco](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2):** DFlash2 draft checkpoint; see its model card for training and upstream algorithm credits.
- **[SGLang](https://github.com/sgl-project/sglang), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [PyTorch](https://github.com/pytorch/pytorch), and [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer):** serving, kernels, tensor runtime, and upstream quantization tooling.

Repository code and patch contributions are licensed under **Apache-2.0**; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Both pinned model cards declare Apache-2.0, but model downloads and dependencies retain their own licenses and notices. No model weights, private request logs, virtual environments, or vendor source trees are distributed in this repository.
