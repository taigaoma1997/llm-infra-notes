# Fundamentals

## Notes

| # | Question | Status |
|---|---|---|
| 01 | How much GPU memory does the KV cache need per token and per request? | planned |
| 02 | Why is prefill compute-bound and decode memory-bound? | planned |
| 03 | How do I read a roofline chart for a given GPU? | planned |
| 04 | What are registers, shared memory, L2 and HBM, and what do they cost to access? | planned |

## Worked example: KV cache size

Bytes per token = 2 (K and V) × layers × KV heads × head dim × bytes per element.

Llama-3-8B has 32 layers, 8 KV heads (grouped-query attention) and head dim 128. In BF16 (2 bytes):

```
2 × 32 × 8 × 128 × 2 B = 131,072 B = 128 KiB per token
8,192-token request      = 1 GiB
```

After about 16 GB of weights, an 80 GB GPU holds a few dozen such long requests at once. The KV cache, not compute, sets the concurrency limit. That is the problem paged KV caches and prefix sharing address.

## Resources

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): roofline and inference chapters
- [Stanford CS336](https://cs336.stanford.edu/): lectures on systems and efficiency
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) Part I: inference concepts, no GPU needed
