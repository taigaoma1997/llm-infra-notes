# 01 · How much memory does the KV cache take, and how many requests fit?

> Examples: Qwen3-0.6B in bf16, nano-vllm @ [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8) · Written: 2026-10-12 · Status: draft

## Question

How much memory does the KV cache take per token, and how many requests fit at once?

## Short answer

Every token keeps a K and a V vector for every KV head in every layer, so its cost is 2 × layers × KV heads × head size × bytes per value. For Qwen3-0.6B in bf16 that is 112 KiB per token, or 28 MiB for one 256-token block. An engine gives the KV cache whatever GPU memory is left after the weights and the temporary memory of its largest step, so the number of requests that fit at once depends on how long they get, not on a batch-size setting.

## How it works

### 1. The formula

Qwen3-0.6B ([config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)): 28 layers, 16 query heads, 8 KV heads, head size 128, bf16 (2 bytes).

| | Size |
|---|---|
| One token, one layer, K only | 8 heads × 128 × 2 bytes = 2 KiB |
| One token, all layers, K and V | 2 × 28 × 2 KiB = **112 KiB** |
| One 256-token block (nano-vllm's unit) | 256 × 112 KiB = **28 MiB** |
| A request with 500 prompt tokens and 300 generated | 800 × 112 KiB ≈ 87.5 MiB; in nano-vllm 4 blocks = 112 MiB |

### 2. What changes it

- **Grouped-query attention.** Only the 8 KV heads are cached, not the 16 query heads. With one KV head per query head, the cost would double, to 224 KiB per token.
- **Data type.** An 8-bit KV cache halves it. nano-vllm keeps the cache in the model's dtype.
- **Tensor parallelism.** With 2 GPUs, each GPU caches 4 of the 8 KV heads: 56 KiB per token and 14 MiB per block on each GPU ([Parallelism](04-parallelism.md)).
- **Block size** does not change the cost per token, only the rounding: a sequence's last block is partly empty, up to 255 unused tokens per sequence with 256-token blocks. mini-sglang uses one token per page and wastes nothing there.

### 3. How nano-vllm decides the size

[`allocate_kv_cache`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L103-L121):

```
blocks = (total × gpu_memory_utilization − used − (peak − current)) ÷ bytes per block
```

- `total × 0.9` is the share of the GPU the engine allows itself.
- `used` is what the GPU already holds, seen from the driver: the weights, the CUDA context, other processes.
- `peak − current` is the extra memory the largest step needs while it runs. The engine measures it just before, in [`warmup_model`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L91-L101), by running one prefill of 16,384 tokens (4 sequences of 4096).

It then allocates one tensor of shape `[2, layers, blocks, 256, KV heads, head size]` and hands each layer its K and V slices. The CUDA graphs are recorded afterwards, so their memory pool has to fit in the remaining 10%.

### 4. What fits

The cache limits how many requests run at once, through their length. With 1024 blocks (262,144 tokens, 28 GiB for this model), requests of 500 + 300 tokens need 4 blocks each by the end, so at most 256 of them can finish together, while `max_num_seqs` allows 512. nano-vllm admits a request as soon as its prompt fits (2 blocks), so 512 get in, and as they grow, it has to preempt. [Experiment 001](../../experiments/001-preemption-1000-requests/) shows when: 171 preempted at step 29, when each needs a third block, and 85 more at step 285.

### 5. A check on my laptop

mini-sglang, on the same RTX 4060 Laptop GPU (8 GB) and the same model, reported a KV cache of 47,008 tokens in 5.02 GiB with `memory_ratio` 0.9 ([mini-sglang notes](../mini-sglang/README.md)). 5.02 GiB ÷ 47,008 = 112 KiB per token, as the formula says.

## Code map

| File and function | Role |
|---|---|
| [`model_runner.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py) `warmup_model` | Measures the peak memory of the largest step |
| `allocate_kv_cache` | Block count, one KV tensor, per-layer slices |

## How I checked

- Computed the sizes from the model config.
- mini-sglang's reported cache size on my laptop matches 112 KiB per token (section 5).
- Not measured yet: how many blocks nano-vllm gets on my 8 GB GPU, and how much of the memory goes to the warmup peak and the CUDA graph pool.

## Open questions

- Where do the 8 GB go on my laptop: weights, KV cache, the peak of the largest step, the CUDA graphs?

## References

- [nano-vllm Note 04](../nano-vllm/04-model-runner-and-cuda-graph.md): startup order
- [Experiment 001](../../experiments/001-preemption-1000-requests/)
