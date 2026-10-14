"""PR #281's W8A16 Triton kernel against bf16 F.linear (cuBLAS), one linear layer at a time. No model is loaded.

For each of Qwen3-0.6B's four linear layers and each M (tokens in the step: the batch size in decode,
all prompt tokens of the step in prefill), times three ways to compute y = x @ W^T:
  bf16           F.linear with bf16 weights, what nano-vllm does today
  int8           the PR's fused kernel: reads int8 weight tiles, converts them on chip, multiplies, applies the scale
  dequant+bf16   no custom kernel: rebuild a full bf16 weight from int8 and scale, then F.linear
Timing: triton.testing.do_bench (warmup, ~100 ms of repeats, L2 flushed before each run, median).
Errors are checked on random weights, so they only indicate the size; see the README for the real model.

Usage: PYTHONPATH=<PR #281 checkout> python bench_kernel.py [--m 1 16 64 ...] [--out results/kernel.csv]
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench

try:
    from nanovllm.layers.w8a16_gemm import w8a16_linear
    from nanovllm.utils.loader import quantize_weight_int8
except ImportError:
    raise SystemExit("nanovllm.layers.w8a16_gemm not found: PYTHONPATH must point to the PR #281 checkout")

# Qwen3-0.6B, one layer: (name, N = output size, K = input size)
SHAPES = [("qkv", 4096, 1024), ("o", 1024, 2048), ("gate_up", 6144, 1024), ("down", 1024, 3072)]
LAYERS = 28
# The PR's kernel uses 16-row tiles for M <= 64 and 64-row tiles above, so both sides of 64 are included.
DEFAULT_M = [1, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384]


def bench_us(fn):
    ms = do_bench(fn, warmup=25, rep=100, quantiles=[0.5])  # with one quantile, triton 3.2 returns a float
    return (ms[0] if isinstance(ms, (list, tuple)) else ms) * 1000


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "kernel.csv"))
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}   torch {torch.__version__}   triton {triton.__version__}")

    rows = []
    for name, N, K in SHAPES:
        w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
        w_int8, scale = quantize_weight_int8(w)                  # the function the PR uses when loading weights
        w_int8, scale = w_int8.cuda(), scale.cuda()
        print(f"\n== {name}: N={N}, K={K}, weight {N * K * 2 / 2**20:.1f} MiB in bf16, {N * K / 2**20:.1f} MiB in int8")
        print(f"{'M':>6} | {'bf16 us':>9} | {'int8 us':>9} | {'dequant+bf16 us':>15} | {'bf16/int8':>9} | "
              f"{'bf16 GB/s':>9} | {'int8 GB/s':>9} | {'bf16 TFLOPS':>11} | {'int8 TFLOPS':>11}")
        for M in args.m:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            t_bf16 = bench_us(lambda: F.linear(x, w))
            t_int8 = bench_us(lambda: w8a16_linear(x, w_int8, scale))
            t_deq = bench_us(lambda: F.linear(x, (w_int8.float() * scale).to(torch.bfloat16)))
            # minimum bytes moved: weight + x + y (int8 also reads one fp32 scale per row)
            bytes_bf16 = 2 * N * K + 2 * M * K + 2 * M * N
            bytes_int8 = N * K + 4 * N + 2 * M * K + 2 * M * N
            flops = 2 * M * N * K
            gbps = lambda b, t: b / (t * 1e-6) / 1e9
            tflops = lambda t: flops / (t * 1e-6) / 1e12
            print(f"{M:>6} | {t_bf16:>9.1f} | {t_int8:>9.1f} | {t_deq:>15.1f} | {t_bf16 / t_int8:>9.2f} | "
                  f"{gbps(bytes_bf16, t_bf16):>9.0f} | {gbps(bytes_int8, t_int8):>9.0f} | "
                  f"{tflops(t_bf16):>11.2f} | {tflops(t_int8):>11.2f}")
            rows.append(dict(shape=name, N=N, K=K, M=M, bf16_us=round(t_bf16, 2), int8_us=round(t_int8, 2),
                             dequant_then_bf16_us=round(t_deq, 2)))

        for M in (16, 4096):  # one M per tile configuration
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            y_int8 = w8a16_linear(x, w_int8, scale).float()
            y_bf16 = F.linear(x, w).float()
            exact_int8 = x.float() @ (w_int8.float() * scale).T   # same int8 weights, computed in fp32
            exact_bf16 = x.float() @ w.float().T                  # original bf16 weights, computed in fp32
            print(f"   relative error, M={M}: kernel {rel_err(y_int8, exact_int8):.1e} (vs. exact result with the int8 weights), "
                  f"quantization {rel_err(y_int8, exact_bf16):.1e} (vs. exact result with the bf16 weights), "
                  f"bf16 F.linear for reference {rel_err(y_bf16, exact_bf16):.1e}")

    print(f"\n== The four linear layers of all {LAYERS} layers, per step (ms)")
    print(f"{'M':>6} | {'bf16':>8} | {'int8':>8} | {'bf16/int8':>9}")
    for M in args.m:
        b = sum(r["bf16_us"] for r in rows if r["M"] == M) * LAYERS / 1000
        q = sum(r["int8_us"] for r in rows if r["M"] == M) * LAYERS / 1000
        print(f"{M:>6} | {b:>8.2f} | {q:>8.2f} | {b / q:>9.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
