"""The GPU's hardware numbers, its achievable memory bandwidth, and what PR #281's kernel compiles to on it.

Prints: device properties from torch and from Triton's driver, the theoretical bandwidth they imply,
a measured device-to-device copy bandwidth (512 MiB, read + write), and for both of the kernel's tile
configurations: threads, registers and shared memory per program, and which PTX instructions appear.

Usage: PYTHONPATH=<PR #281 checkout> python gpu_facts.py
"""
import re
import sys

import torch
import triton

sys.stdout.reconfigure(encoding="utf-8")
from nanovllm.layers.w8a16_gemm import _w8a16_gemm_kernel

p = torch.cuda.get_device_properties(0)
print("== torch device properties")
for k in ["name", "major", "minor", "multi_processor_count", "total_memory", "L2_cache_size", "warp_size",
          "max_threads_per_multi_processor", "regs_per_multiprocessor"]:
    print(f"  {k}: {getattr(p, k)}")

print("== triton driver properties")
props = triton.runtime.driver.active.utils.get_device_properties(0)
print(f"  {props}")
bw = 2 * props["mem_clock_rate"] * 1e3 * props["mem_bus_width"] / 8 / 1e9
print(f"  theoretical bandwidth = 2 x {props['mem_clock_rate'] / 1e3:.0f} MHz x {props['mem_bus_width']} bit / 8 = {bw:.0f} GB/s")
cores = p.multi_processor_count * 128  # 128 FP32 units per SM on this architecture (compute capability 8.9)
print(f"  FP32 peak without Tensor Cores = {cores} units x 2 x {props['sm_clock_rate'] / 1e6:.2f} GHz "
      f"= {cores * 2 * props['sm_clock_rate'] * 1e3 / 1e12:.1f} TFLOPS")

print("== measured copy bandwidth (512 MiB tensor, read + write)")
x = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
y = torch.empty_like(x)
for _ in range(3):
    y.copy_(x)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
times = []
for _ in range(20):
    start.record(); y.copy_(x); end.record(); torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
t = sorted(times)[len(times) // 2] / 1000
print(f"  median {t * 1000:.2f} ms -> {2 * x.numel() / t / 1e9:.0f} GB/s")
del x, y

print("== PR #281's kernel compiled for this GPU (qkv shape: N=4096, K=1024)")
for M, BM, stages in ((1, 16, 3), (16384, 64, 4)):
    N, K = 4096, 1024
    xa = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    s = torch.rand(N, device="cuda", dtype=torch.float32)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, 64))
    k = _w8a16_gemm_kernel[grid](xa, w, s, c, M, N, K, xa.stride(0), xa.stride(1), w.stride(0), w.stride(1),
                                 c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=64, BLOCK_K=64,
                                 num_stages=stages, num_warps=4)
    torch.cuda.synchronize()
    ptx, md = k.asm["ptx"], k.metadata
    print(f"  BLOCK_M={BM}, num_stages={stages}: grid={grid}, threads per program={32 * md.num_warps}, "
          f"registers per thread={k.n_regs}, spills={k.n_spills}, shared memory per program={md.shared} bytes")
    counts = {name: len(re.findall(pat, ptx)) for name, pat in [
        ("mma.sync (Tensor Core)", r"\bmma\.sync"), ("ldmatrix (shared memory -> registers)", r"\bldmatrix"),
        ("cp.async (global -> shared, asynchronous)", r"\bcp\.async\.(?:cg|ca)"), ("ld.global", r"\bld\.global"),
        ("st.global", r"\bst\.global"), ("bar.sync", r"\bbar\.sync"), ("cvt (type conversion)", r"\bcvt\.")]}
    print(f"   PTX instructions in the code: {counts}")
    for pat in (r"mma\.sync[^\n;]*", r"cp\.async\.c[ga][^\n;]*", r"cvt\.rn\.f32\.s8[^\n;]*"):
        m = re.search(pat, ptx)
        if m:
            print(f"   e.g. {m.group(0).strip()}")
