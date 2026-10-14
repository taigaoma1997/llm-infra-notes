# 002 · Does INT8 weight-only quantization (nano-vllm PR #281) make nano-vllm faster on an 8 GB laptop GPU?

> Written: 2026-10-14 · Status: done

## Question

[PR #281](https://github.com/GeeeekExplorer/nano-vllm/pull/281) stores the weights of every linear layer in int8 and multiplies them with a fused Triton kernel (W8A16: 8-bit weights, 16-bit activations). On an RTX 4070 Laptop GPU it reports 7.3% more throughput and 17% more KV cache blocks. Does that hold on an RTX 4060 Laptop GPU, and where exactly does the time go?

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, 24 SMs, driver 617.14 |
| Software | Windows 11; Python 3.10.21, torch 2.6.0+cu124, triton-windows 3.2.0.post21, flash-attn 2.8.3, transformers 5.16.1 ([results/env.txt](results/env.txt)) |
| Code | main at [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8); PR #281 at `c3b7110`, which is based on `bb823b3` |
| Model | Qwen3-0.6B, bf16 |
| Workload | The PR's `bench.py`: `seed(0)`, 64 requests arriving at once, 34,428 prompt tokens and 38,443 generated tokens in total (`ignore_eos`), CUDA graphs on |

## Method

| Script | What it measures |
|---|---|
| [kv_blocks.py](kv_blocks.py) | The KV block count, and the memory terms nano-vllm computes it from |
| [bench_kernel.py](bench_kernel.py) | The PR's kernel alone against bf16 `F.linear` (cuBLAS), for Qwen3-0.6B's four linear-layer shapes and M from 1 to 16,384 (M = tokens in the step: the batch size in decode, the prompt tokens of the step in prefill) |
| [bench_breakdown.py](bench_breakdown.py) | The `bench.py` workload end to end, with every step timed and classified as prefill or decode, and preemptions counted. Three rounds, alternating main bf16, PR bf16 and PR int8 |
| [prefill_steps.py](prefill_steps.py) | The size of every prefill step, by replaying the same workload through the real scheduler without a model |

Each configuration runs in a new process. The scheduler's decisions depend only on token counts, so the three bf16 runs make identical steps, and the replay in `prefill_steps.py` matches the real runs step for step (same step counts, preemptions and prefill tokens).

## Results

### 1. Memory: 17 more KV blocks

From [results/kv_blocks.txt](results/kv_blocks.txt), read when the KV cache is about to be allocated (MiB):

| | used | reserved | current | peak | peak − current | Blocks |
|---|---|---|---|---|---|---|
| main bf16 | 2346 | 1234 | 1165 | 1613 | 448 | 163 |
| PR int8 | 1872 | 760 | 752 | 1201 | 449 | 180 |

nano-vllm sizes the cache as `(total × 0.9 − used − (peak − current)) ÷ 28 MiB`, which gives 163.4 and 180.3 here.

- `current`, almost all weights, is 413 MiB lower. The linear layers hold 440 million parameters: 840 MiB in bf16, 420 MiB in int8 plus 1.3 MiB of scales. The embedding (shared with the LM head, 297 MiB) is not quantized.
- `peak − current`, the temporary memory of the largest step (one 16,384-token prefill), is the same: activations stay in bf16.
- `used − reserved`, everything outside PyTorch, is 1112 MiB in both. The other 61 MiB of the drop in `used` is memory PyTorch keeps cached without any tensor in it (`reserved − current`: 69 MiB in bf16, 8 MiB in int8), which the formula counts as used. That is why int8 gets 17 blocks rather than the 15 its weights alone would give.

### 2. The kernel alone: faster for decode, slower for large prefill

Full tables: [results/kernel_run1.txt](results/kernel_run1.txt), [results/kernel_run2.txt](results/kernel_run2.txt). One step through the four linear layers of all 28 layers:

| M | bf16 (ms) | int8 (ms) | bf16 / int8 |
|---|---|---|---|
| 1 | 7.77 | 5.88 | 1.32 |
| 16 | 7.86 | 6.25 | 1.26 |
| 64 | 8.52 | 7.08 | 1.20 |
| 128 | 9.86 | 9.29 | 1.06 |
| 256 | 16.49 | 16.40 | 1.01 |
| 1024 | 53.00 | 57.98 | 0.91 |
| 2048 | 107.46 | 123.83 | 0.87 |
| 4096 | 211.96 | 253.73 | 0.84 |
| 16384 | 858.10 | 1770.51 | 0.48 |

- **Small M is limited by memory bandwidth.** From M = 1 to 64, bf16 time hardly changes: reading the weights dominates. Halving the weight bytes helps, but only in two of the four layers. At M = 1, qkv (N = 4096) is 1.85× faster and gate_up (N = 6144) 1.54×, while o and down (N = 1024) are not faster at all and reach only about 50 GB/s. With 16-row by 64-column output tiles, a layer with N = 1024 gives 16 thread blocks for 24 SMs, and each walks the whole K dimension (2048 or 3072) alone. cuBLAS reaches about 100 GB/s on the same shapes.
- **Large M is limited by compute**, and there cuBLAS wins: bf16 stays at 16–18 TFLOPS, the PR's kernel drops to 7–9 at M = 16,384 and takes twice as long. It has one fixed 64 × 64 × 64 tile and converts int8 to bf16 inside the loop.
- **Fusing matters.** Rebuilding the bf16 weight first and then calling `F.linear` (third column of the full tables) is about 4× slower than bf16 at M = 1, because the weight is read and written several times. At M = 16,384 it is within 3% of bf16, twice as fast as the PR's kernel.
- **Accuracy.** On random weights, the kernel's own error (1.6–1.7 × 10⁻³ relative) equals bf16's rounding error; quantization adds about 8 × 10⁻³, five times that.

### 3. End to end: no reliable gain

From [results/breakdown.csv](results/breakdown.csv), mean of three runs (min–max):

| | main bf16 | PR bf16 | PR int8 |
|---|---|---|---|
| Total | 43.98 s (43.34–44.73) | 44.56 s (43.95–45.54) | 43.30 s (42.73–44.02) |
| Prefill | 4.47 s, 26 steps, 52,438 tokens | 4.54 s, same steps | 5.96 s, 20 steps, 48,572 tokens |
| Decode | 39.41 s, 1492 steps | 39.90 s, same steps | 37.25 s, 1233 steps |
| Preemptions | 34 | 34 | 24 |
| Tokens recomputed after preemption | 18,010 | 18,010 | 14,144 |

Decode time per step by batch size:

| Batch | main bf16 | PR int8 | |
|---|---|---|---|
| 1–8 | 11.47 ms | 9.64 ms | −16% |
| 9–16 | 19.87 | 18.12 | −9% |
| 17–32 | 31.27 | 32.08 | +3% |
| 33–64 | 37.27 | 41.60 | +12% |

- int8 is 0.68 s (1.5%) faster on average, but the ranges overlap; two identical bf16 runs differed by 1.4 s. The PR's bf16 path is as fast as main, within the same noise.
- **Decode: −2.16 s, mostly from capacity.** The 17 extra blocks cut preemptions from 34 to 24, more sequences decode together (31.1 per step on average instead of 25.7), and the same tokens take 17% fewer steps. Small-batch steps are faster, as the kernel test predicts; large-batch steps are slower. A decode step with 30–60 sequences mostly reads K/V, which int8 does not shrink, and with 180 blocks more context is in flight per step. I have not verified that last point.
- **Prefill: +1.49 s.** Each prefill token costs 44% more (85.3 against 122.8 µs).

The `bench.py` script itself, run once per configuration, gave 45.38 s (main), 46.19 s (PR bf16) and 45.54 s (PR int8) ([results/bench_py_*.json](results/)).

### 4. Why prefill got slower: two large steps

Prefill steps average about 2000 tokens, where the kernel is only 13–16% slower. The average hides the distribution ([results/prefill_steps.txt](results/prefill_steps.txt)):

| | First two prefill steps | Every other prefill step |
|---|---|---|
| main bf16 | 15,705 and 16,364 tokens | 24 steps of 31 to 2,359 tokens, 20,369 tokens in total |
| PR int8 | the same two steps | 18 steps of 176 to 2,359 tokens, 16,503 tokens in total |

All 64 requests arrive together, so the first two steps fill the 16,384-token budget; the later ones are preempted sequences coming back and stragglers. Interpolating the kernel table over every step gives 2.78 s of linear-layer time for bf16 and 4.43 s for int8, a difference of 1.65 s against the 1.49 s measured. The two large steps alone account for about 1.8 s; fewer recomputed tokens give back 0.2 s.

## Conclusion

On this laptop, PR #281 saves 413 MiB of weights and turns them into 17 more KV blocks, but it does not make this workload measurably faster: 43.30 s against 43.98 s, within run-to-run noise. Decode gains 2.2 s, mostly because more KV blocks mean fewer preemptions and larger batches, not because each step is faster; prefill loses 1.5 s in two 16k-token steps, where the kernel takes twice as long as cuBLAS. The +7.3% reported on an RTX 4070 Laptop GPU is plausible there because that GPU was more memory-starved (98 blocks growing to 115, +17%, against 163 to 180, +10% here), so the capacity effect is larger.

Using the kernel only where it wins would change the result. If the steps above 4096 tokens rebuilt the bf16 weight and used cuBLAS, which is within 3% of bf16 at that size, the int8 prefill would cost about 2.56 s of linear-layer time instead of 4.43 s, and the total would drop to roughly 41.4 s, about 6% faster than main. That is an estimate, not a measurement: the temporary bf16 weight would also raise the warmup's peak memory and could cost a block or two.

## Caveats

- One GPU, one model, one workload. Here 61% of the prefill tokens fall in two maximum-size steps because every request arrives at once; with requests arriving over time, prefill steps are smaller and the kernel's prefill penalty shrinks.
- Laptop GPU clocks vary: identical runs differ by about 3%.
- The kernel's error was measured on random weights. On the real model, greedy decoding of a 600-token prompt matched bf16 for 44 of 64 tokens and then picked a different, equally plausible number; the bf16 runs in [#219](../../notes/nano-vllm/issues/219-same-step-prefix-sharing.md#side-finding-why-two-correct-runs-drift-apart) diverged earlier from smaller changes, so the divergence point says little about accuracy. A per-position comparison (teacher forcing) was not done.
- `winfix.py` swaps NCCL for gloo on Windows. A single GPU never calls a collective, so it does not affect the numbers.

## Reproduce

```bash
git -C nano-vllm fetch origin pull/281/head:pr-281
git -C nano-vllm worktree add ../pr-281 pr-281
NANO_VLLM_MAIN=/path/to/nano-vllm NANO_VLLM_PR=/path/to/pr-281 PYTHON=/path/to/python bash run.sh
```

`NANO_VLLM_MAIN` is a checkout at `bb823b3`. [run.sh](run.sh) writes everything in `results/`; the nine end-to-end runs take about 12 minutes.
