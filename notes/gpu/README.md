# GPU programming

> Status: started (Q&A pages)

Some basics already came up while reading nano-vllm: what a kernel is, grids, blocks and threads, a naive and a tiled matmul, Triton versus CUDA, the kernels `torch.compile` generates, and how to read the kernels inside a recorded CUDA graph. They are in my Chinese Q&A page [QA-cuda-graph.md](../nano-vllm/QA-cuda-graph.md), section 5.

## Q&A (in Chinese)

The questions I asked while learning the basics, with answers. They started from reading the Triton kernel in nano-vllm [PR #281](../nano-vllm/issues/281-int8-weight-only-quantization.md), and the numbers come from my RTX 4060 Laptop GPU ([experiment 002](../../experiments/002-int8-weight-only-pr281/), `gpu_facts.py`).

| Topic | Page | Covers |
|---|---|---|
| GPU hardware | [QA-gpu.md](QA-gpu.md) | SMs, L2 and memory; what is inside an SM (warp schedulers, CUDA cores, Tensor Cores, registers, shared memory); threads, warps, blocks and how many blocks fit on an SM; the memory hierarchy; the rules for writing a kernel that follow from the hardware; when a kernel is limited by memory bandwidth or by compute; this GPU's numbers, including a measured copy bandwidth of about 196 GB/s |
| Triton | [QA-triton.md](QA-triton.md) | What Triton is and how it compiles; the vector-add example; the common operations (`program_id`, `arange`, pointers, `load` with masks, `dot`, `constexpr`) and the launch syntax; what PR #281's kernel compiles to (`cp.async`, `ldmatrix`, `mma.sync` on Tensor Cores) with its registers and shared memory; how many programs fit on an SM, and how that explains the kernel's speed |

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
