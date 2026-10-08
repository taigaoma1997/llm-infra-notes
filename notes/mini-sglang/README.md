# mini-sglang

> Upstream: [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) · Commit I read: [`9a91cfa`](https://github.com/sgl-project/mini-sglang/commit/9a91cfafe754aa85daee49998176275667eb58f2) · Status: in progress (started 2026-10-05)

A compact version of SGLang, about 5,000 lines of Python. It keeps SGLang's main ideas: a radix-tree prefix cache, chunked prefill, overlap scheduling, CUDA Graph for decode, and tensor parallelism. I read it after [nano-vllm](../nano-vllm/), so my notes compare the two.

## How I run it

mini-sglang runs on Linux only: sgl-kernel, FlashInfer and mini-sglang's own CUDA kernels are compiled on the machine with `nvcc` the first time they are used. My laptop runs Windows, so everything runs in WSL2.

| | |
|---|---|
| Machine | Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9) |
| Linux | WSL2, Ubuntu 24.04 |
| Driver | NVIDIA 617.14 on Windows (WSL uses the Windows driver) |
| Python | 3.12, one [uv](https://docs.astral.sh/uv/) virtual environment shared by all checkouts |
| Packages | torch 2.9.1+cu128, flashinfer-python 0.7.0.post1, sgl-kernel 0.3.21, apache-tvm-ffi 0.1.14.post1, transformers 4.57.3 |
| CUDA Toolkit | 12.8 inside WSL, for `nvcc` |
| Model | Qwen3-0.6B |

### Where things live

The project is at `~/LLM/mini-sglang` in the WSL file system, not under `/mnt/c`: files there cross the Windows–Linux boundary on every access, which is slow for git, Python imports and kernel compilation. VSCode opens the folder through its WSL extension.

The layout is the same as for nano-vllm: three git worktrees of one clone, each with a fixed role, plus a lab folder for tests, drafts and logs.

```
~/LLM/mini-sglang/
├── learning/    branch learning: my comments and tracing; holds the shared .git; the editable install points here
├── dev/         one branch per upstream fix, fix/<issue>-<name>
├── upstream/    branch upstream-main, unmodified
├── lab/         tools, one folder per issue, scratch experiments
├── .venv/       Python 3.12 (uv)
└── activate.sh  activates .venv and puts nvcc on PATH
```

### Install

Inside WSL:

```bash
# CUDA Toolkit 12.8 from NVIDIA's WSL repository (toolkit only; WSL uses the Windows driver)
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-8

# uv, three worktrees, one virtual environment
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/LLM/mini-sglang && cd ~/LLM/mini-sglang
git clone https://github.com/sgl-project/mini-sglang learning
cd learning
git branch -m main upstream-main            # tracks origin/main
git switch -c learning
git worktree add ../upstream upstream-main
git worktree add --detach ../dev upstream-main
cd ..
uv venv --python=3.12 .venv && source .venv/bin/activate
uv pip install -e "learning[dev]" torch-c-dlpack-ext
```

There is no conda environment. In each new terminal, `source ~/LLM/mini-sglang/activate.sh` enters `.venv` and puts `nvcc` on `PATH`.

### First run

The upstream unit tests that need no GPU pass: `python -m pytest --no-cov tests/core` in `learning` gives 6 passed.

End to end, I ran the offline engine (`minisgl.llm.LLM`) on two chat prompts from nano-vllm's `example.py`, with `temperature=0.6` and `max_tokens=256`, all other settings at their defaults:

| | First run | Second run |
|---|---|---|
| From start to both outputs | 175 s | 16 s |
| Kernels compiled | 5 FlashInfer kernels: decode, prefill, rope, silu_and_mul, sampling | none, read from the cache (15 MB) |
| Output | 155 tokens (ended at EOS) and 256 tokens (hit `max_tokens`) | the same tokens |

- The attention backend is chosen from the GPU: FlashInfer (`fi`) on this card, compute capability 8.9. Hopper GPUs get FlashAttention for prefill and FlashInfer for decode.
- With 6.92 GiB free and `memory_ratio` 0.9, the KV cache got 47,008 tokens (5.02 GiB). Each token gets its own slot (page size 1), where nano-vllm uses blocks of 256 tokens.
- It captures 23 CUDA graphs, for batch sizes 1, 2, 4, then every 8 up to 160.
- Both runs give the same tokens even with `temperature=0.6`, because the engine seeds torch with 42 at startup.

## How I reproduce and test fixes

The same approach as for [nano-vllm](../nano-vllm/README.md#how-i-reproduce-and-test-fixes): one test script per issue, run against a chosen checkout by a small launcher, in one process per checkout. Two things are different here:

- **`minisgl` is a namespace package**: it has no `__init__.py`, so Python merges every `minisgl/` folder it finds on `sys.path` into one package. The editable install keeps the `learning` checkout on `sys.path`, so putting another checkout first is not enough: a file that exists only in `learning` would still be imported from there, without an error. The launcher removes every other copy from `sys.path` and checks that `minisgl.__path__` has exactly one entry.
- **The scheduler simulation keeps the real scheduler** and replaces only the model step. A fake engine returns one fixed token per request; message handling, prefill admission, chunked prefill, the radix cache, page allocation and the overlap loop are mini-sglang's own code. It loads no weights and runs a scenario in seconds. I checked it against the real engine on two scenarios, which cover chunked prefill, waiting for memory, eviction and a prefix-cache hit, with overlap scheduling on and off. The step-by-step traces (which requests run, how many tokens each has, free pages) were identical.

## Notes

| # | Question | Status |
|---|---|---|
| 01 | What does mini-sglang add on top of nano-vllm, and why? | planned |
| 02 | How does the radix cache find and reuse a shared prefix? | planned |
| 03 | How does overlap scheduling hide CPU work behind GPU work? | planned |

## Issues I reproduced

Write-ups will live in [issues/](issues/), together with a [backlog](issues/README.md#backlog) of the open issues and PRs I can reproduce on one GPU. The status of each one is tracked in [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md).

## Resources

- [Mini-SGLang announcement (LMSYS)](https://www.lmsys.org/blog/2025-12-17-minisgl/)
- [Structure of Mini-SGLang](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md): processes, data flow, and what each package does
- [Features of Mini-SGLang](https://github.com/sgl-project/mini-sglang/blob/main/docs/features.md): command-line options for each feature
