# GPU programming

> Status: planned

## Plan

1. [GPU-Puzzles](https://github.com/srush/GPU-Puzzles): CUDA concepts through small puzzles in Numba (runs on Colab).
2. [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles): Triton from first principles up to Flash Attention (runs in an interpreter, no GPU needed).
3. Triton softmax and matmul, each with an Nsight Compute profile and a short explanation of the bottleneck.
4. Triton FlashAttention-2 forward pass, compared with PyTorch SDPA at several sequence lengths.

## Exercises

| # | Kernel | Result | Status |
|---|---|---|---|
| 01 | softmax | | planned |
| 02 | matmul | | planned |

## Resources

- [GPU MODE lectures](https://github.com/gpu-mode/lectures): 1 to 5 (basics), 8 (performance checklist), 12 (Flash Attention), 14 (Triton)
- [LeetGPU](https://leetgpu.com/): kernel problems with GPUs in the browser
- *Programming Massively Parallel Processors* (PMPP), chapters 1 to 6
