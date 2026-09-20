#!/usr/bin/env python3
"""Compare native NVFP4 causal multi-token attention against dequantized reference."""
import json
import math

import flashinfer
import torch


def check_workspace_gather():
    from sglang.kernels.ops.kvcache.nvfp4_dequant import gather_nvfp4_to_fp8

    torch.manual_seed(1729)
    checks = []
    for count in (0, 1, 67, 4097, 260003):
        slots, heads, dim = max(count, 257), 4, 256
        packed = [
            torch.randint(
                0, 256, (slots, heads, dim // 2), dtype=torch.uint8, device="cuda"
            )
            for _ in range(2)
        ]
        scales = [
            torch.randint(
                0, 127, (slots, heads, dim // 16), dtype=torch.uint8, device="cuda"
            ).view(torch.float8_e4m3fn)
            for _ in range(2)
        ]
        globals_ = [
            torch.tensor([value], dtype=torch.float32, device="cuda")
            for value in (0.03125, 2.0)
        ]
        indices = torch.randint(0, slots, (count,), dtype=torch.int64, device="cuda")
        outputs = [
            torch.empty((count, heads, dim), dtype=torch.float8_e4m3fn, device="cuda")
            for _ in range(2)
        ]
        gather_nvfp4_to_fp8(
            packed[0],
            packed[1],
            scales[0],
            scales[1],
            globals_[0],
            globals_[1],
            indices,
            outputs[0],
            outputs[1],
        )
        torch.cuda.synchronize()
        for data, sf, scale, actual in zip(packed, scales, globals_, outputs):
            if count == 0:
                continue
            expected = (
                flashinfer.nvfp4_kv_dequantize(
                    data[indices].reshape(-1, dim // 2),
                    sf.view(torch.uint8)[indices].reshape(-1, dim // 16),
                    scale,
                    output_dtype=torch.bfloat16,
                )
                .to(torch.float8_e4m3fn)
                .reshape_as(actual)
            )
            a, b = actual.view(torch.uint8), expected.view(torch.uint8)
            equal_nan = ((a & 127) == 127) & ((b & 127) == 127)
            different = (a != b) & ~equal_nan
            mismatches = different.count_nonzero().item()
            assert mismatches == 0, (
                count,
                scale.item(),
                mismatches,
                a[different][:16].tolist(),
                b[different][:16].tolist(),
            )
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gather_nvfp4_to_fp8(
            packed[0],
            packed[1],
            scales[0],
            scales[1],
            globals_[0],
            globals_[1],
            indices,
            outputs[0],
            outputs[1],
        )
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - before
        assert extra == 0, (count, extra)
        checks.append(
            {
                "gather_tokens": count,
                "reference": "bit-exact finite BF16-to-FP8; NaNs equivalent",
                "temporary_tensor_bytes": extra,
            }
        )
    return checks


def main():
    torch.manual_seed(20260919)
    gather_checks = check_workspace_gather()
    page_size, kv_heads, q_heads, dim = 64, 4, 24, 256
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    results = []
    for lengths, q_len in [
        ([67], 1),
        ([8], 8),
        ([67, 131], 4),
        ([67, 131], 6),
        ([67, 131], 8),
        ([1029], 8),
    ]:
        max_pages = math.ceil(max(lengths) / page_size)
        num_pages = max_pages * len(lengths)
        shape = (num_pages * page_size, kv_heads, dim)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(
            (len(lengths) * q_len, q_heads, dim), device="cuda", dtype=torch.bfloat16
        )
        k_scale = torch.tensor([0.0125], device="cuda", dtype=torch.float32)
        v_scale = torch.tensor([0.0075], device="cuda", dtype=torch.float32)
        k4, ks = flashinfer.nvfp4_kv_quantize(k.reshape(-1, dim), k_scale)
        v4, vs = flashinfer.nvfp4_kv_quantize(v.reshape(-1, dim), v_scale)
        kd = flashinfer.nvfp4_kv_dequantize(
            k4, ks, k_scale, output_dtype=torch.bfloat16
        ).reshape(shape)
        vd = flashinfer.nvfp4_kv_dequantize(
            v4, vs, v_scale, output_dtype=torch.bfloat16
        ).reshape(shape)
        packed = tuple(
            x.view(num_pages, page_size, kv_heads, dim // 2).transpose(1, 2)
            for x in (k4, v4)
        )
        scales = tuple(
            x.view(torch.float8_e4m3fn)
            .view(num_pages, page_size, kv_heads, dim // 16)
            .transpose(1, 2)
            for x in (ks, vs)
        )
        tables = torch.arange(num_pages, device="cuda", dtype=torch.int32).view(
            len(lengths), max_pages
        )
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        mask = (
            torch.tensor(
                [[[(1 << (row + 1)) - 1, 0] for row in range(q_len)] for _ in lengths],
                dtype=torch.uint16,
                device="cuda",
            )
            if q_len > 1
            else None
        )
        output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=packed,
            kv_cache_sf=scales,
            workspace_buffer=workspace,
            block_tables=tables,
            seq_lens=lens,
            max_seq_len=max(lengths),
            bmm1_scale=float(k_scale.item()) / math.sqrt(dim),
            bmm2_scale=float(v_scale.item()),
            q_len_per_req=q_len,
            out_dtype=torch.bfloat16,
            mask=mask,
        )
        refs = []
        for batch, length in enumerate(lengths):
            start = batch * max_pages * page_size
            keys = (
                kd[start : start + length]
                .float()
                .repeat_interleave(q_heads // kv_heads, dim=1)
                .transpose(0, 1)
            )
            values = (
                vd[start : start + length]
                .float()
                .repeat_interleave(q_heads // kv_heads, dim=1)
                .transpose(0, 1)
            )
            queries = q[batch * q_len : (batch + 1) * q_len].float().transpose(0, 1)
            scores = queries @ keys.transpose(-1, -2) / math.sqrt(dim)
            causal = (
                torch.arange(length, device="cuda")[None, :]
                <= (length - q_len + torch.arange(q_len, device="cuda"))[:, None]
            )
            scores.masked_fill_(~causal, -torch.inf)
            refs.append((scores.softmax(-1) @ values).transpose(0, 1))
        reference = torch.cat(refs)
        error = output.float() - reference
        relative_rms = (error.square().mean() / reference.square().mean()).sqrt().item()
        max_error = error.abs().max().item()
        if (
            not torch.isfinite(output).all()
            or relative_rms > 0.012
            or max_error > 0.025
        ):
            raise AssertionError((lengths, q_len, relative_rms, max_error))
        # The same fixed geometry must remain correct under CUDA graph replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=packed,
                kv_cache_sf=scales,
                workspace_buffer=workspace,
                block_tables=tables,
                seq_lens=lens,
                max_seq_len=max(lengths),
                bmm1_scale=0.0125 / math.sqrt(dim),
                bmm2_scale=0.0075,
                q_len_per_req=q_len,
                out_dtype=torch.bfloat16,
                mask=mask,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, output, rtol=0.015, atol=0.004)
        results.append(
            {
                "lengths": lengths,
                "query_tokens": q_len,
                "relative_rms_error": relative_rms,
                "max_abs_error": max_error,
                "cuda_graph_replay": "pass",
            }
        )
    print(
        json.dumps(
            {"workspace_gather": gather_checks, "native_attention": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
