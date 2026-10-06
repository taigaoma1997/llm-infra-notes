# nano-vllm

> Upstream: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) · Commit I read: [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8) · Status: in progress

A small, readable engine that keeps the core ideas of vLLM: continuous batching, a paged KV cache with prefix caching, CUDA Graph for decode, and tensor parallelism.

## How I ran it

Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), driver 566.24. Conda environment `learn-vllm`: Python 3.10.21, torch 2.6.0+cu124 (CUDA 12.4), triton-windows 3.2.0, flash-attn 2.8.3, transformers 5.16.1.

```powershell
git clone https://github.com/GeeeekExplorer/nano-vllm; cd nano-vllm
git checkout bb823b3
pip install -e .
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
$env:USE_LIBUV = "0"
python example.py
```

Two things fail on Windows before the model even loads (upstream [#261](https://github.com/GeeeekExplorer/nano-vllm/issues/261)):

- PyTorch on Windows has no NCCL. In `ModelRunner.__init__`, I pass `"gloo"` instead of `"nccl"` to `dist.init_process_group` when `platform.system() == "Windows"`.
- The TCP store needs libuv, which PyTorch on Windows also lacks. Setting `USE_LIBUV=0` before running avoids it.

The first run spends about a minute in the warmup pass, mostly compiling `torch.compile` and Triton kernels.

## How I reproduce and test fixes

Everything for nano-vllm lives in one project folder: three checkouts plus a `lab` folder for tests, drafts and logs. Other repos I study get the same layout. The three checkouts are git worktrees of one clone, each with a fixed role, and I switch branches rather than making a new folder per issue:

| Checkout | Branch | Role |
|---|---|---|
| upstream | `upstream-main` | Unmodified upstream: how the original behaves |
| dev | `fix/<issue>-<name>`, one per issue | The fix I will send upstream, nothing else |
| learning | `learning` | My tracer, comments, and experiment-only options |

Tests and notes for each issue live in one local folder, outside all three checkouts. A small launcher runs any test script against a chosen checkout. It puts that checkout first on `sys.path` and runs each version in its own process, one after another: Python caches modules by name, so two versions of the same package cannot be compared in one process. On Windows it also swaps NCCL for gloo at runtime, so the upstream code stays untouched.

```mermaid
flowchart LR
  S["One test script<br/>for the issue"] --> N{"Launcher:<br/>which checkout?"}
  N -->|upstream| U["Before the fix"]
  N -->|dev| D["After the fix"]
  N -->|learning| L["With tracing"]
  N -->|ab| AB["Before and after,<br/>one command"]
```

Two kinds of test script:

- **Scheduling and KV-cache bugs** (who runs, block allocation, preemption, prefix caching, chunked prefill): drive `Scheduler` and `BlockManager` directly with fake tokens. No GPU, a few seconds per run. Prompts given as lengths get distinct tokens, so they never share a prefix by accident; prompts given as token lists can share one on purpose. Each prefill step prints how many blocks are still free, which shows whether requests shared blocks. The [#274 write-up](issues/274-kv-cache-exhausted-assert.md) shows an early version.
- **Everything else** (model numerics, CUDA Graph, kernels, and crashes inside the model): the real engine. To shrink the KV cache on unmodified code, lower `gpu_memory_utilization`: 0.36 gave about 5 blocks on my 8 GB GPU, depending on free memory.

Each new issue starts from a template: the issue text fetched from GitHub, a checklist from reproduce to PR, one script of each kind, and folders for drafts and logs. The tools are local for now.

Pitfalls I hit:

- `LLMEngine` silently drops keyword arguments it does not know. An option that only exists in my learning branch (like a KV block cap) is ignored on upstream without an error, so a cross-version test can quietly run a different scenario. Settings that exist in every version, like `gpu_memory_utilization`, are safe.
- On Windows, `bash` in cmd or PowerShell starts WSL, not Git Bash; this repo's scripts need Git Bash.
- With torch 2.6 on Windows, `torch.compile` can fail with `PermissionError: [WinError 5]` while renaming a folder in its Triton cache (`%TEMP%\torchinductor_<user>\triton\0\`). It happens when the compile cache (`fxgraph\`) has an entry whose Triton kernels are missing from `triton\0\`: `TritonBundler.read_and_emit` creates the target folder, then moves a temporary folder onto it with `os.replace`, which Windows does not allow. From then on, every run fails the same way. Fixed in torch 2.7 ([pytorch#146481](https://github.com/pytorch/pytorch/pull/146481)). On 2.6, setting `TORCHINDUCTOR_BUNDLE_TRITON_INTO_FX_GRAPH_CACHE=0` before importing torch skips that code path (my launcher does this on Windows), and deleting `fxgraph\` repairs the cache itself.

## Architecture

<!-- Skeleton of the top-level loop. Check it against the commit you read, then extend it. -->

```mermaid
flowchart LR
  A["LLM.generate"] --> B["LLMEngine.step"]
  B --> C["Scheduler.schedule<br/>prefill batch or decode batch"]
  C --> D["ModelRunner.run"]
  D --> E["Scheduler.postprocess<br/>append tokens, finish sequences"]
  E --> B
  C -.-> F["BlockManager<br/>allocate, share by prefix hash, free"]
```

## Notes

| # | Question | Status |
|---|---|---|
| 01 | What happens to a request between `generate()` and the returned text? | planned |
| 02 | How does the scheduler choose prefill or decode, and when does it preempt? | planned |
| 03 | How are KV cache blocks allocated, shared by prefix hash, and freed? | planned |
| 04 | What does ModelRunner prepare for each step, and where is CUDA Graph used? | planned; much of it is already in the [#190 write-up](issues/190-cuda-graph-block-tables.md#background-how-one-decode-step-runs) |
| 05 | How does tensor parallelism split the model and keep workers in sync? | planned |

## Issues I reproduced

Write-ups live in [issues/](issues/). The status of each one is tracked in [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md).

## Q&A (in Chinese)

The questions I asked while studying each topic, with answers, diagrams and self-checks. These are my study pages, so they are in Chinese.

| Topic | Page | Covers |
|---|---|---|
| CUDA graph | [QA-cuda-graph.md](QA-cuda-graph.md) | Graph buffers, how a decode batch runs, CUDA graph and why one per batch size, what a recorded graph contains, GPU kernels from a naive matmul to FlashAttention, how buffers, graph nodes and kernels connect, a self-quiz. Started from [#190](issues/190-cuda-graph-block-tables.md) |

## Resources

- [Understanding LLM Inference Engines: Inside Nano-vLLM](https://neutree.ai/blog/nano-vllm-part-1)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm), for comparison with the real engine
