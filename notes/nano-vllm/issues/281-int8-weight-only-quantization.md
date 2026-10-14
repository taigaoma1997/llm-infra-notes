# [nano-vllm PR #281] INT8 weight-only (W8A16) quantization with a fused Triton GEMM

- Upstream PR: https://github.com/GeeeekExplorer/nano-vllm/pull/281 (open, no comments as of 2026-10-14)
- Status: tested on an 8 GB laptop GPU; no comment posted yet
- Code tested: the PR at `c3b7110`, against main at [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8), which the PR is based on
- Numbers and scripts: [experiment 002](../../../experiments/002-int8-weight-only-pr281/)
- Q&A in Chinese: [QA-quantization](../QA-quantization.md)
- Written: 2026-10-14

## TL;DR

- The PR stores every linear layer's weights in int8 and multiplies them with its own Triton kernel. It runs on Windows with triton-windows, and its kernel is as accurate as bf16 `F.linear`; quantization itself adds about 0.8% relative error to a layer's output.
- It saves 413 MiB of weights on Qwen3-0.6B, and nano-vllm turns that memory into 17 more KV blocks (163 to 180). The description's "~1.5 GB to ~0.75 GB" does not match: the linear weights are 840 MiB in bf16.
- On the PR's own benchmark it is not measurably faster on my RTX 4060 Laptop GPU: 43.30 s against 43.98 s for main, within run-to-run noise. The description reports +7.3% on an RTX 4070 Laptop GPU.
- Splitting the time explains why. Decode gets 2.2 s faster, mostly because the extra blocks mean fewer preemptions and larger batches; each decode step is only faster at small batch sizes. Prefill gets 1.5 s slower, almost all of it in two 16k-token steps, where the kernel takes twice as long as cuBLAS.
- Two kernel changes would likely make it pay off: hand large prefill steps to cuBLAS after rebuilding the bf16 weight (estimated ~6% faster than main on this benchmark), and give the narrow layers (o_proj, down_proj) more parallelism at small batch sizes. Neither is tested.

## What I learned

- Quantization != faster -> Based on my experiments on int8, faster decode (as tokens are less), slower prefill (tokens are more). We need to split this into details. 
- How int impacts the decodes? -> Only when the batch is small (decode token is small) it is faster. The faster time comes from the less evicited blocks, fewer steps due to more blocks available -> Getting more memory for blocks is already a speedup! Not necessarily due to the quantization. 
- Int8 quantization does not change the peak memory usage -> since int8 only change weights, we still need activations in bf16. 
- o/down is not speed up in int8 -> only need 16 threads, but GPU has 24 SM, so it does not fully utilize the GPU resources -> Also reminds me that speeds up should also depends on the GPU and models! That's how infra works!
- Need to run multiple times to remove the noise on time. 

## Background: what the PR changes

W8A16 means 8-bit weights and 16-bit activations. When the checkpoint is loaded, each linear layer's weight is quantized per output row: the row's largest absolute value divided by 127 becomes its scale, and every weight in the row becomes a rounded integer from −128 to 127 ([`quantize_weight_int8`](https://github.com/AliMizan/nano-vllm/blob/c3b7110a57ab603048774d1b52a434670f92d3ed/nanovllm/utils/loader.py#L14-L30)). The four projections of every layer (qkv, o, gate_up, down) are quantized; the embedding, the LM head (which shares the embedding's weight in Qwen3-0.6B) and the RMSNorms are not.

At run time, every linear layer calls [`w8a16_linear`](https://github.com/AliMizan/nano-vllm/blob/c3b7110a57ab603048774d1b52a434670f92d3ed/nanovllm/layers/w8a16_gemm.py#L54-L97), whatever the number of tokens M in the step. Its [kernel](https://github.com/AliMizan/nano-vllm/blob/c3b7110a57ab603048774d1b52a434670f92d3ed/nanovllm/layers/w8a16_gemm.py#L11-L51) is a tiled matrix multiply: each thread block loads a tile of int8 weights, converts it to bf16 on chip, accumulates `tl.dot` in fp32, and multiplies by the per-row scale at the end. The weights are read from GPU memory once, as int8. It uses 16 × 64 × 64 tiles for M ≤ 64 and 64 × 64 × 64 above.

Decode reads every weight once per step, so at small M the step time is the time to read the weights, and halving their size should help. At large M the time goes into arithmetic, where the int8 bytes no longer matter and the kernel has to compete with cuBLAS.

## What I checked

All on an RTX 4060 Laptop GPU (8 GB, 24 SMs), Windows 11, torch 2.6.0+cu124, triton-windows 3.2.0, Qwen3-0.6B in bf16. Details and raw output: [experiment 002](../../../experiments/002-int8-weight-only-pr281/).

### Output

Greedy decoding (sampling replaced by argmax) of a 600-token prompt, 64 tokens:

- main and the PR without quantization: identical tokens.
- The PR with int8: the first 44 tokens are identical; token 45 is a different number ("is 42" instead of "is 395"), and the rest follows from it.

Quantization shifts the logits slightly, and greedy decoding flips at the first near-tie. The same prompt drifted after 13 or 22 tokens in [#219](219-same-step-prefix-sharing.md#side-finding-why-two-correct-runs-drift-apart) from nothing more than a different attention kernel in bf16, so the divergence point says little about accuracy. On random weights, the kernel's own error (1.7 × 10⁻³ relative) equals bf16's rounding error, and quantization adds about 8 × 10⁻³.

### Memory

nano-vllm sizes the KV cache as `(total × 0.9 − used − (peak − current)) ÷ 28 MiB` ([Note 04](../04-model-runner-and-cuda-graph.md#1-startup)). The three terms, read just before the cache is allocated:

| | used | current | peak − current | Blocks |
|---|---|---|---|---|
| main, bf16 | 2346 MiB | 1165 | 448 | 163 |
| PR, int8 | 1872 MiB | 752 | 449 | 180 |

- `current` is almost all weights and drops by 413 MiB, as expected from 440 million linear-layer parameters (840 MiB in bf16, 420 MiB in int8 plus 1.3 MiB of scales).
- `peak − current`, the temporary memory of the largest step, does not change: activations are still bf16.
- `used` drops by another 61 MiB, which turns out to be memory PyTorch keeps cached with no tensor in it (`memory_reserved() − current` is 69 MiB in bf16, 8 MiB in int8). The part of `used` outside PyTorch is 1112 MiB in both.

### The kernel alone

The four linear layers of all 28 layers, one step:

| M | bf16 | int8 | bf16 / int8 |
|---|---|---|---|
| 1 | 7.77 ms | 5.88 ms | 1.32 |
| 64 | 8.52 | 7.08 | 1.20 |
| 256 | 16.49 | 16.40 | 1.01 |
| 2048 | 107.46 | 123.83 | 0.87 |
| 16384 | 858.10 | 1770.51 | 0.48 |

- **Decode (M ≤ 64)**: qkv and gate_up are 1.4–1.85× faster; o_proj and down_proj (N = 1024) are not faster at all and reach only about 50 GB/s against cuBLAS's 100. With 64-column tiles, N = 1024 gives 16 thread blocks for 24 SMs, and each walks the whole K dimension (2048 or 3072) alone.
- **Prefill (M ≥ 1024)**: slower, twice as slow at M = 16,384, where cuBLAS keeps 16–18 TFLOPS and the kernel drops to 7–9.
- Rebuilding the bf16 weight and calling `F.linear` instead is 4× slower than bf16 at M = 1 but within 3% of it at M = 16,384, so it beats the PR's kernel there.

### End to end

The PR's `bench.py` workload (64 requests at once, 34,428 prompt tokens, 38,443 generated), with every step timed. Mean of three runs:

| | main, bf16 | PR, bf16 | PR, int8 |
|---|---|---|---|
| Total | 43.98 s (43.34–44.73) | 44.56 s | 43.30 s (42.73–44.02) |
| Prefill | 4.47 s, 26 steps | 4.54 s | 5.96 s, 20 steps |
| Decode | 39.41 s, 1492 steps | 39.90 s | 37.25 s, 1233 steps |
| Preemptions | 34 | 34 | 24 |

- **No reliable gain.** int8 is 1.5% faster on average, but identical runs differ by 3%.
- **Decode, −2.2 s, comes mostly from capacity.** With 180 blocks instead of 163, there are 10 fewer preemptions, 31 sequences decode per step on average instead of 26, and the same tokens take 17% fewer steps. Per step, int8 is 16% faster with 1–8 sequences and 9% faster with 9–16, but 3% slower with 17–32 and 12% slower with 33–64. My guess for the last part, not verified: large-batch decode steps mostly read K/V, which int8 does not shrink, and with more blocks more context is in flight.
- **Prefill, +1.5 s, comes from two steps.** Replaying the workload through the scheduler shows that the first two prefill steps hold 15,705 and 16,364 tokens (all 64 requests arrive at once) and the rest hold at most 2,359. Interpolating the kernel table over every step predicts 1.65 s more linear-layer time for int8, against 1.49 s measured; the two large steps alone account for about 1.8 s.

So the reported +7.3% most likely comes from the extra KV blocks rather than faster steps. The RTX 4070 Laptop GPU in the description had 98 blocks growing to 115 (+17%), against 163 to 180 here (+10%): it was short of KV cache to begin with, so 17 more blocks prevent more preemptions there.

## Changes bundled with the quantization

The PR also changes code that runs without quantization:

| Change | Effect here |
|---|---|
| NCCL-or-gloo backend and `USE_LIBUV=0` on Windows | Same fix as [PR #267](https://github.com/GeeeekExplorer/nano-vllm/pull/267) for [#261](https://github.com/GeeeekExplorer/nano-vllm/issues/261) |
| `warmup_model` runs the largest prefill twice and measures the peak only in the second run | Same block count as main (163) |
| `@torch.compile(dynamic=True)` on both RMSNorm functions | Same greedy output and same speed as main, within noise |
| `LLMEngine.exit` checks `hasattr(self, "model_runner")` | Needed by the PR's `bench.py`, which calls `llm.exit()` itself: `exit` is also registered with `atexit`, so on main the second call raises `AttributeError` when the program ends |
| `import os` twice in `model_runner.py` | None |

Each could be its own small PR; the maintainer tends to merge small, focused changes.

## Ideas that could make it pay off (not tested)

- **Hand large M to cuBLAS.** Above about 4096 tokens, rebuild the bf16 weight from int8 and the scales and call `F.linear`. In this workload only the two large prefill steps would take that path, and the estimate is about 41.4 s, 6% faster than main. The temporary bf16 weight (12 MiB for gate_up, more if it goes through fp32) would also raise the warmup's measured peak and could cost a block or two.
- **More thread blocks for narrow layers at small M.** Smaller column tiles or a split-K reduction for o_proj and down_proj, so that 16 thread blocks no longer leave a third of the SMs idle.
- **Grouped tile order for large M**, as in Triton's matmul tutorial, so neighbouring thread blocks reuse the same input tiles from L2.

## Thoughts

- A quantization result depends on the workload mix. W8A16 speeds up memory-bound decode and does nothing for compute-bound prefill, so the same PR can win with short prompts and long answers and lose with long prompts and short answers.
- Some of the gain from quantization is capacity, not speed: smaller weights leave room for more KV blocks, and on a GPU that is short of KV cache, that matters more than the kernel.
- Step counts are as informative as times. Because scheduling depends only on token counts, the three bf16 runs made exactly the same 1518 steps, which isolated the timing noise, and a scheduler-only replay reproduced every step.
- Two traps in benchmarking this: the PR's `bench.py --compare` starts subprocesses, which in my setup imported a different nano-vllm checkout (where `quantization="int8"` is silently ignored); and a single run per configuration cannot separate a 1–2% effect from noise.

## Open questions

- The bf16 decode kernels reach only 100–130 GB/s. What does a plain copy reach on this GPU, and how far is that from its specification?
- How large is the quantization error on the real model, measured per position with teacher forcing rather than by where greedy output diverges?
- Are large-batch int8 decode steps slower because more K/V is in flight? Comparing steps with the same total context length would tell.
