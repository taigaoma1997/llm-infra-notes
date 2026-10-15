# GPU programming

> Status: started (Q&A pages)

Some basics already came up while reading nano-vllm: what a kernel is, grids, blocks and threads, a naive and a tiled matmul, Triton versus CUDA, the kernels `torch.compile` generates, and how to read the kernels inside a recorded CUDA graph. They are in my Chinese Q&A page [QA-cuda-graph.md](../nano-vllm/QA-cuda-graph.md), section 5.

## Q&A (in Chinese)

The questions I asked while learning the basics, with answers. They started from reading the Triton kernel in nano-vllm [PR #281](../nano-vllm/issues/281-int8-weight-only-quantization.md), and the numbers come from my RTX 4060 Laptop GPU ([experiment 002](../../experiments/002-int8-weight-only-pr281/), `gpu_facts.py`).

| Topic | Page | Covers |
|---|---|---|
| GPU hardware | [QA-gpu.md](QA-gpu.md) | SMs, L2 and memory; what is inside an SM (warp schedulers, CUDA cores, Tensor Cores, registers, shared memory); threads, warps, blocks and how many blocks fit on an SM; the memory hierarchy; the rules for writing a kernel that follow from the hardware; when a kernel is limited by memory bandwidth or by compute; this GPU's numbers, including a measured copy bandwidth of about 196 GB/s |
| Triton | [QA-triton.md](QA-triton.md) | What Triton is and how it compiles; the vector-add example; the common operations (`program_id`, `arange`, pointers, `load` with masks, `dot`, `constexpr`) and the launch syntax; what PR #281's kernel compiles to (`cp.async`, `ldmatrix`, `mma.sync` on Tensor Cores) with its registers and shared memory; how many programs fit on an SM, and how that explains the kernel's speed |
| Courses | [QA-courses.md](QA-courses.md) | Well-known online lectures on GPUs, kernels and Triton: why each one, which lectures to watch, and in what order (the list below) |

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

Lectures, in the order I plan to watch them. Why each one, in Chinese: [QA-courses.md](QA-courses.md).

1. Stanford [CS336](http://cs336.stanford.edu/spring2025/) (Spring 2025, [videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)): lecture 5 (GPUs) and lecture 6 (kernels, Triton). Later: 7 and 8 (parallelism), 10 (inference).
2. [GPU MODE lectures](https://github.com/gpu-mode/lectures) ([YouTube](https://www.youtube.com/@GPUMODE)): 4 (compute and memory architecture) and 14 (Triton) first, then Triton-Puzzles (step 2 of the plan).
3. More GPU MODE lectures:
   - 3 (CUDA for Python programmers)
   - 8 (performance checklist)
   - 23 (Tensor Cores)
   - 34 (low-bit Triton kernels; read again with PR #281's kernel)
   - 29 (Triton internals)
   - 12 (Flash Attention)
   - 35 (SGLang performance)

To go deeper:

- *Programming Massively Parallel Processors* (PMPP), chapters 1 to 6, with the GPU Computing lectures by Izzat El Hajj, one of its authors: [2020 and 2021](https://www.youtube.com/playlist?list=PLZrjSW9GrEZE07f8gLECc0tfJwNDm75RW), [2026](https://www.youtube.com/playlist?list=PLZrjSW9GrEZE_uL0qSjc7dzFwnI2AjPm1). GPU MODE lecture 2 recaps chapters 1 to 3.
- Stanford [CS149](https://gfxcourses.stanford.edu/cs149/fall25) Parallel Computing ([2023 videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)): the lecture on modern multi-core processors, and lecture 7 (GPU architecture and CUDA). These explain why the hardware is built this way.
- MIT [6.5940](https://hanlab.mit.edu/courses/2026-fall-65940) TinyML and Efficient Deep Learning Computing: the quantization lectures.

To look things up and to practice:

- [GPU Glossary](https://modal.com/gpu-glossary/readme) (Modal): GPU terms, linked to each other
- [LeetGPU](https://leetgpu.com/): kernel problems with GPUs in the browser
- [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda): CUDA kernels with notes, including notes on the GPU MODE lectures (`cuda-mode/`)
