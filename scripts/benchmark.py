#!/usr/bin/env python3
"""Single-request FP8/NVFP4 comparison; generated Python is compiled, never run."""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Huihui-Qwen3.8-27B-Abliterated-Gittensor-NVFP4"
PUBLIC_RUNTIME_FIELDS = (
    "context_length", "dtype", "quantization", "kv_cache_dtype",
    "mem_fraction_static", "max_running_requests", "max_total_tokens",
    "chunked_prefill_size", "page_size", "attention_backend",
    "decode_attention_backend", "prefill_attention_backend",
    "speculative_algorithm", "speculative_num_draft_tokens",
    "speculative_dflash_block_size", "speculative_dflash_window_size",
    "speculative_attention_mode", "speculative_draft_model_quantization",
    "speculative_draft_kv_cache_dtype", "max_mamba_cache_size",
    "mamba_ssm_dtype", "mamba_scheduler_strategy", "mamba_backend",
    "cuda_graph_config", "cuda_graph_max_bs_decode", "disable_cuda_graph",
    "disable_radix_cache", "enable_multimodal", "language_only", "version",
)
PUBLIC_META_FIELDS = (
    "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "cached_tokens", "cached_tokens_details", "num_retractions",
    "queue_time", "e2e_latency", "decode_throughput", "forward_entry_time",
    "prefill_finished_time", "api_server_dispatch_finish_ts",
    "request_received_ts", "request_finished_ts", "response_sent_to_client_ts",
    "spec_accept_rate", "spec_accept_length", "spec_verify_ct",
    "spec_num_correct_drafts", "spec_num_proposed_drafts",
    "spec_accepted_drafts", "spec_proposed_drafts", "spec_accept_histogram",
    "spec_correct_drafts_histogram",
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def server_url(value):
    parts = urlsplit(value.strip())
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path.rstrip("/") not in ("", "/v1")
    ):
        raise argparse.ArgumentTypeError("use an HTTP(S) server root, optionally ending in /v1")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

CODE_TASK = """Ignore the archive unless it is relevant. Write a complete Python module implementing a bounded LRU cache with TTL. Use class LRUCache(capacity, ttl, clock), with injected monotonic clock, get(key, default=None), put(key, value), and __len__. Expiry is inclusive (now >= deadline); successful get makes an entry most recent; replacing a key refreshes its TTL; expired entries must not evict live entries. Reject nonpositive capacity or TTL. Include deterministic unittest cases for expiry boundaries, LRU eviction, overwrites, and expired-entry cleanup. Return only Python code, no markdown or explanation."""
RECALL_KEYS = ["alder", "birch", "cedar", "dogwood", "elm", "fir"]
POSITIONS = [0.01, 0.10, 0.25, 0.50, 0.75, 0.95]
RECALL_TASK = (
    "Return only a JSON object mapping these six archive secret keys to their exact values: "
    + ", ".join(RECALL_KEYS)
    + ". Do not infer values from other records."
)


def request(url, path, body=None, timeout=1200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def generate(url, ids, max_tokens):
    body = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "stream": True,
    }
    started = time.perf_counter()
    first = last = None
    first_count = last_count = 0
    final = None
    done = False
    with request(url, "/generate", body) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            meta = event.get("meta_info", {})
            count = meta.get("completion_tokens", 0)
            if count > last_count:
                now = time.perf_counter()
                if first is None:
                    first, first_count = now, count
                last, last_count = now, count
            final = event
    elapsed = time.perf_counter() - started
    if not done or final is None or first is None:
        raise RuntimeError("Incomplete generation stream")
    meta = final["meta_info"]
    if meta.get("prompt_tokens") != len(ids):
        raise RuntimeError(
            f"Prompt token mismatch: {len(ids)} vs {meta.get('prompt_tokens')}"
        )
    return {
        "prompt_tokens": len(ids),
        "completion_tokens": meta.get("completion_tokens"),
        "ttft_s": round(first - started, 6),
        "total_s": round(elapsed, 6),
        "decode_tok_s": (
            round((last_count - first_count) / (last - first), 3)
            if last > first
            else None
        ),
        "first_stream_token_count": first_count,
        "output": final.get("text", ""),
        "meta_info": {key: meta[key] for key in PUBLIC_META_FIELDS if key in meta},
    }


def make_prompt(tokenizer, filler, depth, case):
    task = CODE_TASK if case == "code" else RECALL_TASK
    template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Follow the user's instructions precisely."},
            {
                "role": "user",
                "content": "Reference archive begins.\n__ARCHIVE_PLACEHOLDER__\nReference archive ends.\n\n"
                + task,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    left, right = template.split("__ARCHIVE_PLACEHOLDER__")
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    n = depth - len(prefix) - len(suffix)
    if n < 0:
        raise ValueError(f"Depth {depth} is smaller than the {case} chat template")
    archive = filler[:n].copy()
    if len(archive) != n:
        raise RuntimeError("Filler too short")
    expected = {}
    offsets = {}
    if case == "recall":
        if n < 4096:
            raise ValueError("Recall requires at least 4096 archive tokens")
        for key, fraction in zip(RECALL_KEYS, POSITIONS):
            value = (
                hashlib.sha256(f"kv-comparison-2026:{depth}:{key}".encode())
                .hexdigest()[:12]
                .upper()
            )
            expected[key] = value
            marker = tokenizer.encode(
                f"\nARCHIVE SECRET: {key} = {value}\n", add_special_tokens=False
            )
            offset = int(n * fraction)
            if offset + len(marker) > n:
                raise ValueError("Recall marker does not fit the requested depth")
            archive[offset : offset + len(marker)] = marker
            offsets[key] = len(prefix) + offset
    ids = prefix + archive + suffix
    if len(ids) != depth:
        raise ValueError(f"Requested {depth} prompt tokens, constructed {len(ids)}")
    return ids, expected, offsets


def score_recall(output, expected):
    try:
        content = output.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        answer = json.loads(content)
    except (ValueError, TypeError):
        answer = None
    correct = [
        key
        for key, value in expected.items()
        if isinstance(answer, dict) and answer.get(key) == value
    ]
    return {
        "correct": len(correct),
        "total": len(expected),
        "all_correct": len(correct) == len(expected),
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--url", type=server_url, default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=MODEL, help="Existing local tokenizer directory")
    parser.add_argument("--depths", nargs="+", type=positive_int, default=[8192, 32768, 122000])
    parser.add_argument(
        "--cases", nargs="+", choices=["code", "recall"], default=["code", "recall"]
    )
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--max-tokens", type=positive_int, default=1536, help="Code output cap; recall always uses 192")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if not args.model.is_dir():
        parser.error("--model must be an existing local tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    filler = []
    for start in range(0, 16000, 1024):
        records = "".join(
            f"Operational record {i:06d}: service={['billing', 'search', 'storage', 'gateway'][i % 4]} zone={i % 13} status=healthy latency_ms={11 + i % 97} checksum={hashlib.sha256(str(i).encode()).hexdigest()[:8]}.\n"
            for i in range(start, start + 1024)
        )
        filler.extend(tokenizer.encode(records, add_special_tokens=False))
        if len(filler) >= max(args.depths):
            break
    with request(args.url, "/server_info") as response:
        info = json.load(response)
    context_length = info.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        parser.error("server_info must provide a positive context_length")
    for depth in args.depths:
        for case in args.cases:
            output_limit = 192 if case == "recall" else args.max_tokens
            if depth + output_limit > context_length:
                parser.error(f"{case}: depth {depth} plus output cap {output_limit} exceeds context {context_length}")
            try:
                make_prompt(tokenizer, filler, depth, case)
            except (ValueError, RuntimeError) as error:
                parser.error(str(error))
    results = {
        "arm": args.arm,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methodology": {
            "temperature": 0,
            "thinking": False,
            "recall_max_output_tokens": 192,
            "max_output_tokens": args.max_tokens,
            "repeats": args.repeats,
            "concurrency": 1,
            "decode_rate": "(last streamed completion count - first streamed completion count) / (last token-event time - first token-event time)",
            "cache": "Flush before each case/depth; repetition 1 cold, subsequent repetitions warm.",
            "filler": "Deterministic diverse synthetic operational records; exact requested token occupancy.",
            "code_task": CODE_TASK,
            "recall_task": RECALL_TASK,
            "recall_positions": POSITIONS,
            "caveats": "Small synthetic sample, not general accuracy equivalence or a broad coding benchmark. TPS is workload dependent. TTFT includes HTTP overhead and prompt processing.",
        },
        "runtime_settings": {key: info[key] for key in PUBLIC_RUNTIME_FIELDS if key in info},
        "tokenizer": (
            "heswithme/Huihui-Qwen3.8-27B-Abliterated-Gittensor-Style-NVFP4-RTX5090"
            if args.model.resolve() == MODEL.resolve() else "custom-local-tokenizer"
        ),
        "redaction": "Runtime settings and per-request metadata are allowlisted. Local model paths, environment/configuration dumps, request IDs and weight-version identifiers are not stored.",
        "runs": [],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as output:
        output.write(json.dumps(results, indent=2) + "\n")
    # Warm kernels without polluting any measured prefix.
    warm = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with the word ready."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    generate(args.url, tokenizer.encode(warm, add_special_tokens=False), 16)
    for depth in args.depths:
        for case in args.cases:
            ids, expected, offsets = make_prompt(tokenizer, filler, depth, case)
            digest = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            with request(args.url, "/flush_cache", {}) as response:
                response.read()
            for repeat in range(args.repeats):
                record = {
                    "case": case,
                    "requested_depth": depth,
                    "repeat": repeat + 1,
                    "cache_condition": "cold" if repeat == 0 else "warm",
                    "input_ids_sha256": digest,
                }
                try:
                    record.update(
                        generate(
                            args.url, ids, 192 if case == "recall" else args.max_tokens
                        )
                    )
                    if case == "recall":
                        record.update(
                            expected=expected,
                            marker_token_offsets=offsets,
                            recall=score_recall(record["output"], expected),
                        )
                    else:
                        code = re.sub(
                            r"^```(?:python)?\s*|\s*```$", "", record["output"].strip()
                        )
                        try:
                            # Syntax check only: never execute generated code.
                            compile(code, "<generated>", "exec")
                            record["python_syntax_valid"] = True
                        except SyntaxError as error:
                            record["python_syntax_valid"] = False
                            record["syntax_error"] = str(error)
                except Exception as error:
                    # Keep server error bodies and potentially private paths out of result files.
                    record["error"] = type(error).__name__
                    if isinstance(error, urllib.error.HTTPError):
                        record["http_status"] = error.code
                    results["runs"].append(record)
                    out.write_text(json.dumps(results, indent=2) + "\n")
                    raise
                results["runs"].append(record)
                out.write_text(json.dumps(results, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if key
                            not in (
                                "output",
                                "meta_info",
                                "expected",
                                "marker_token_offsets",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
