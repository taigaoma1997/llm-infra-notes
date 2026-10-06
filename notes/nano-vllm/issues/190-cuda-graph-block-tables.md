# [nano-vllm #190] CUDA graph replay fails once a sequence grows past max_model_len

- Upstream issue: https://github.com/GeeeekExplorer/nano-vllm/issues/190 · Same bug, reported earlier: [#106](https://github.com/GeeeekExplorer/nano-vllm/issues/106) · Open PRs that mention it: [#191](https://github.com/GeeeekExplorer/nano-vllm/pull/191), [#258](https://github.com/GeeeekExplorer/nano-vllm/pull/258), [#263](https://github.com/GeeeekExplorer/nano-vllm/pull/263)
- Status: investigating. I understand the bug from the code and have looked inside the recorded graphs; reproducing it is next.
- Commit read: `bb823b3`, unmodified upstream
- Written: 2026-10-03 · Updated: 2026-10-05
- Study Q&A (Chinese), with every question I asked along the way: [QA-cuda-graph.md](../QA-cuda-graph.md)

<!-- Also add a row to CONTRIBUTIONS.md and keep its status in sync with this page. -->

## TL;DR

- **The bug:** with CUDA graph on (`enforce_eager=False`), decode crashes with `RuntimeError: The expanded size of the tensor (16) must match the existing size (17)` as soon as any sequence in the batch grows past `max_model_len` (4096 by default).
- **Why:** CUDA graph replays GPU work recorded at startup, and that work reads fixed buffers. The `block_tables` buffer has `max_model_len / 256` = 16 columns, so it fits sequences of up to 4096 tokens. Nothing stops a sequence from growing longer: neither the scheduler nor `SamplingParams` checks `max_model_len`. A 4097-token sequence needs a 17th column, and the copy into the buffer fails.
- **Only the CUDA graph path is affected.** Prefill, `enforce_eager=True`, and batches of more than 512 sequences build their tensors fresh every step. `example.py` sets `enforce_eager=True`, so it never hits this.
- **Next:** reproduce it on my GPU, then weigh the fix directions and the open PRs.

## What I learned

- What cuda_graph looks like? -> print the results and checked
- Difference on batch size -> the kernel changes wr.t. batch size. for batch = 1 or 16, the mat_mul change from gemv to cutlass, attention changed from 4 pieces to more pieces, 
- How long sequence is split -> flash_fwd_splitkv + combine
- Understanding of block table, cuda graph, kernel -> cuda graph records a series of kernel actions, and at each action, kernels fetches data based on blcok table info. 


## Background: how one decode step runs

To follow this bug I first had to understand how a decode step turns a batch of sequences into GPU work. The example below is used throughout: three running sequences, A with 600 tokens, B with 300, and C with 100.

### A batch is computed together

In a decode step, every sequence in the batch adds one token, and all of them go through the model in the same forward pass. Each GPU operation is launched once for the whole batch.

Most layers see a plain matrix: one row per sequence, so `[3, 1024]` here, with no padding. Each row is computed independently. Positions differ per row (599, 299, 99), and the rotary embedding uses each row's own position. Each new token's K and V are written to its own slot in the KV cache.

Attention is the only layer that looks at history, and the histories differ in length and live in different KV blocks of 256 tokens. One call to FlashAttention handles all three sequences, so it gets two extra inputs:

```
KV blocks per sequence (256 tokens each):
A: 600 tokens → 3 blocks [5, 9, 12]
B: 300 tokens → 2 blocks [7, 3]
C: 100 tokens → 1 block  [4]

block_tables =             context_lens =
[[5,  9, 12],    ← A       [600,
 [7,  3, -1],    ← B        300,
 [4, -1, -1]]    ← C        100]
```

- **Why pad?** A GPU tensor must be a rectangle, so shorter rows are padded to the longest row.
- **Does padding cost compute?** No. For each sequence, attention reads only `ceil(context_len / 256)` entries of its row, so the `-1` entries are never read. `-1` is just a value that can never be a valid block ID.
- **Does it cost memory?** Almost none: 4 bytes per entry. Even the whole CUDA graph buffer (512 × 16) is 32 KB, while one KV block is about 28 MB (2 × 28 layers × 256 tokens × 8 KV heads × 128 dims × 2 bytes).

### What CUDA graph does

CUDA graph changes how the work is launched, not what is computed.

- **Eager:** Python launches each GPU operation one by one. A decode step is 395 operations at batch size 1, as I counted from a recorded graph ([below](#inside-a-recorded-graph)): 14 per layer times 28 layers, plus a few more. Each launch costs the CPU a few microseconds, while each operation does very little work (one token per sequence). So the GPU spends much of its time waiting for the next launch.
- **CUDA graph:** at startup, nano-vllm records the whole list of operations, including the memory addresses they read and write. Each decode step then runs the whole list with one `graph.replay()` call.
- **The price:** the recording fixes every address and shape. Each step must copy its inputs into the same buffers that were used during recording, and those buffers cannot change size.

Details that follow from this:

- **One graph per batch size.** Shapes are part of the recording, so nano-vllm records 36 graphs, for batch sizes 1, 2, 4, 8, 16, 32 and so on up to 512. A batch is padded up to the next recorded size: 3 sequences use the graph for 4, and 17 use the graph for 32. Empty seats get `context_lens = 0`, so attention reads nothing for them, but they still go through every matrix multiply, on whatever token and position earlier steps left in those seats. Their results are thrown away, and `slot_mapping = -1` stops them from writing to the KV cache: an empty seat may compute garbage, but it never writes it anywhere that matters.
- **Largest first, one memory pool.** The graphs are recorded from largest to smallest and share one memory pool, so the smaller ones reuse the memory the largest one claimed.
- **Decode only.** A decode step is small and repeats thousands of times, so launch overhead dominates. A prefill step works on hundreds or thousands of tokens, so the GPU work dominates, and its size changes every time.
- **`compute_logits` runs outside the graph.** It runs eagerly after the replay.

### One decode step in code

```mermaid
flowchart TD
  S["Scheduler.schedule<br/>pick the batch; add a block<br/>when a sequence crosses 256 tokens"] --> P["prepare_decode<br/>one token per sequence, positions,<br/>context_lens, block_tables padded with -1"]
  P --> Q{"enforce_eager, or more<br/>than 512 sequences?"}
  Q -->|yes| E["Eager: launch each GPU op from Python.<br/>Attention reads the fresh block_tables."]
  Q -->|no| C["Copy the inputs into the fixed buffers.<br/>The block_tables buffer has 16 columns."]
  C --> F{"Does the table fit?"}
  F -->|yes| R["graph.replay():<br/>all recorded ops in one call"]
  F -->|"no: a sequence passed max_model_len"| X["RuntimeError (#190)"]
  classDef bad fill:#ffe0e0,stroke:#cc0000,color:#000
  class X bad
```

| Step | Code |
|---|---|
| Each sequence keeps its own list of block IDs | [`Sequence.block_table`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/sequence.py#L28) |
| The scheduler adds a block when a sequence crosses a 256-token boundary, with no length limit | [`may_append`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L106-L108), called from [`schedule`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L57-L73) |
| The batch's inputs are built on the CPU, with `block_tables` padded to the longest row | [`prepare_decode`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L172-L188), [`prepare_block_tables`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L123-L127) |
| They go on a global "context" that attention reads, because the model's entry point only takes `input_ids` and `positions` | [`set_context`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/utils/context.py) |
| Eager, or copy into the buffers and replay | [`run_model`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L196-L212) |
| At startup: the buffers are created and one graph is recorded per batch size, with the context pointing at the buffers | [`capture_cudagraph`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L223-L257) |
| Attention reads `block_tables` and `context_lens` from the context | [`attention.py` L71-L74](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L71-L74) |

The key link is in `capture_cudagraph`: during recording, the context points at the buffers. So the recorded attention reads the buffer's memory, and each replay reads the same memory.

## The bug

The buffer's width is set once, at startup ([model_runner.py L227, L232](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L227-L232)):

```python
max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size   # 4096 / 256 = 16
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)               # 512 rows × 16 columns
```

Before each replay, the step's table is copied into the top-left corner of that buffer ([L210](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L210)):

```python
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
```

Follow one sequence as it grows:

| Sequence length | Blocks needed | Columns in this step's table | Fits in 16 columns? |
|---|---|---|---|
| 4095 | 16 | 16 | ✅ |
| 4096 | 16 | 16 | ✅ |
| 4097 | 17 | 17 | ❌ `RuntimeError` |

That matches the error in #190: `Target sizes: [32, 16]. Tensor sizes: [32, 17]`, meaning 32 sequences, a buffer 16 entries wide, and a table 17 entries wide. #106 shows `64` versus `65`, so that user had set `max_model_len` to 16384. A comment there names the cause: `max_tokens` was 20000 while `max_model_len` was the default 4096.

**One long sequence fails the whole batch.** The table is one rectangle, as wide as its longest row, so a single sequence past the limit makes the copy fail for the entire step. The exception propagates out of `generate()`, which collects results in a local variable and returns only at the end. Every request in that call loses its output, including the ones that had already finished.

`max_model_len` is used in only two places: to size the warmup batch and to size this buffer. It is not a limit. Apart from this buffer, the model could handle longer sequences: the rotary embedding covers 40960 positions, and an 8 GB GPU holds over 40K tokens of KV cache.

**Why not use a wider tensor at runtime?** The recorded attention kernel holds the buffer's address and its row width. A new tensor would live at another address that the recording does not know about. Using a wider table means recording the graphs again, which takes seconds.

### Two claims in the issue to check

- "The buffer is not cleared before copying." Columns beyond this step's width keep values from earlier steps. Does attention ever read them, given that it stops at `context_lens`?
- "Allocate one extra column." Does that fix the bug, or only move the crash from 4097 tokens to 4353?

## Inside a recorded graph

To see what a graph actually holds, I turned on CUDA graph debug mode before nano-vllm recorded its graphs (`enable_debug_mode()`, by swapping in a `CUDAGraph` subclass, with nano-vllm's code unchanged), then wrote each graph out with `debug_dump()`. The dump lists every recorded node: the kernel and its launch configuration `<<<grid, threads per block, shared memory>>>`. Settings: `max_num_seqs=16`, so 5 graphs were recorded, for batch sizes 1, 2, 4, 8 and 16.

**The graph for batch size 1 has 395 nodes: 394 kernels and 1 memory copy.** After the embedding lookup, every layer is the same 14 kernels:

| # in layer | Kernel | What it does |
|---|---|---|
| 1 | `triton_per_fused_..._rsqrt` | `input_layernorm`: RMSNorm, fused into one kernel by `torch.compile` |
| 2 | `gemvx` | qkv projection (a matrix-vector product, since there is one token) |
| 3, 4 | `triton_per_fused_..._rsqrt` | `q_norm` (grid 16: one per query head) and `k_norm` (grid 8: one per KV head) |
| 5, 6 | `triton_poi_fused_..._cat` | rotary embedding on q, then on k |
| 7 | `store_kvcache_kernel` | write the new token's K and V into the KV cache |
| 8, 9 | `flash_fwd_splitkv_kernel`, `flash_fwd_splitkv_combine_kernel` | attention, split into 4 chunks, then the chunks combined |
| 10 | `gemvx` | output projection |
| 11 | `triton_per_fused_..._rsqrt` | `post_attention_layernorm`, with the residual add |
| 12 | `cutlass::Kernel2` | gate and up projections |
| 13 | `triton_poi_fused_mul_silu` | SiLU and multiply |
| 14 | `gemvx` | down projection |

The counts add up: 28 layers × 14 = 392, plus the embedding and the final norm makes 394 kernels. The extra node copies the result into the `outputs` buffer. The RMSNorm kernel appears 113 times (28 × 4 + 1), and `gemvx` 84 times (28 × 3).

**Batch size 16 is not the same graph with bigger launches.** Its graph has 367 nodes, and the kernels themselves change:

| | Batch size 1 | Batch size 16 |
|---|---|---|
| Matrix multiplies | `gemvx` (matrix × vector) for qkv, output and down | `cutlass::Kernel2` (matrix × matrix) for all four |
| Attention | grid `{1, 4, 8}`: 4 splits × 8 KV heads, then a combine kernel | grid `{1, 16, 8}`: 16 sequences × 8 KV heads, no split, no combine kernel |
| `q_norm`, `k_norm` | `triton_per_...` | `triton_red_...`, a different reduction strategy |
| Elementwise kernels | grids of 1, 8, 12 blocks | grids of 16, 128, 192 blocks |

The missing combine kernel accounts for the difference: 395 − 28 = 367. So a graph really is tied to one batch size: the kernel choice, not only the launch sizes, depends on it.

The attention split is the split-KV idea from [below](#questions-for-later-decode-performance), done automatically by FlashAttention: at batch size 1 there are only 8 blocks of work (one per KV head), too few to fill the GPU, so each is split in 4. At batch size 16 there is enough work without splitting. The split count is part of the recorded launch, so it is fixed when the graph is recorded.

## How buffers, graph nodes and kernels fit together

Three things are easy to mix up here. **Kernels** are the code that runs on the GPU. A **CUDA graph** is a recorded list of kernel launches. The **buffers**, such as `block_tables`, are plain blocks of GPU memory. They connect through addresses: each recorded launch holds the addresses of the memory its kernel will read and write.

### What one node holds

```mermaid
flowchart TD
  N["<b>One graph node</b><br/>attention, graph for batch size 1"]
  N --> K["<b>① Kernel</b><br/>flash_fwd_splitkv_kernel"]
  N --> C["<b>② Launch config</b><br/>grid {1,4,8} · 128 threads"]
  N --> A["<b>③ Arguments</b><br/>GPU memory addresses"]
  A --> B1["block_tables<br/>0xB000 · 16 per row"]
  A --> B2["context_lens<br/>0xC000"]
  A --> B3["KV cache<br/>0xA000"]
  classDef node fill:#fff3c4,stroke:#b8860b,color:#000
  classDef code fill:#e6f4ea,stroke:#34a853,color:#000
  classDef cfg fill:#f1f3f4,stroke:#5f6368,color:#000
  classDef mem fill:#e8f0fe,stroke:#4a6fa5,color:#000
  class N node
  class K code
  class C cfg
  class A,B1,B2,B3 mem
```

| Part | In this node | Fixed when | Can it change later? |
|---|---|---|---|
| ① Kernel | `flash_fwd_splitkv_kernel`, FlashAttention's GPU code | when the graph is recorded | no |
| ② Launch config | grid `{1,4,8}` (1 block of queries × 4 splits × 8 KV heads), 128 threads per block, 80 KB of shared memory | when the graph is recorded | no |
| ③ Arguments | the addresses of `block_tables` (rows 16 entries wide), `context_lens`, the KV cache, and `q` from earlier nodes | when the graph is recorded | no |
| The data at those addresses | this step's block numbers and lengths, the cached K and V | — | yes, every step |

A node holds no data and no code, only these three things. The addresses are illustrative; the launch configuration is the one measured for attention in the graph for batch size 1. Because the addresses and the row width are fixed when the graph is recorded, a buffer can be neither replaced nor widened. That is the root of #190.

### At startup: recording the graphs

```mermaid
sequenceDiagram
    participant CPU
    participant MEM as GPU memory
    participant G as CUDA graph
    CPU->>MEM: ① allocate the KV cache
    CPU->>MEM: ② allocate the graph buffers
    CPU->>G: ③ run the model while recording
    Note right of G: each kernel launch<br/>is stored as a node,<br/>not run
    Note over CPU,G: ④ repeat ③ for batch sizes 16, 8, 4, 2, 1
```

| Step | What happens | Code |
|---|---|---|
| ① | `allocate_kv_cache` allocates one tensor for every layer's K and V. Each attention layer keeps a view of it, so its address is fixed from now on | [model_runner.py L103-L121](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L103-L121) |
| ② | `capture_cudagraph` allocates the graph buffers: `input_ids`, `positions`, `slot_mapping`, `context_lens` (512 each), `block_tables` (512 × 16) and `outputs` (512 × 1024) | [L227-L233](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L227-L233) |
| ③ | `set_context` points attention at the buffers. One warm-up pass runs, then the same pass runs inside `torch.cuda.graph(...)`, which stores each kernel launch as a node instead of running it | [L240-L243](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L240-L243) |
| ④ | The loop repeats ③ for every batch size, largest first. All graphs share one memory pool | [L238-L246](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L238-L246) |

### Every decode step: replaying a graph

```mermaid
sequenceDiagram
    participant CPU
    participant BUF as Graph buffers
    participant G as CUDA graph<br/>(runs its kernels)
    participant KV as KV cache
    CPU->>BUF: ① copy this step's inputs
    CPU->>G: ② replay(), no data
    G->>BUF: ③ read inputs
    G->>KV: ④ read and write K, V
    G->>BUF: ⑤ write outputs
    BUF->>CPU: ⑥ read outputs, compute logits
```

| Step | What happens | Code |
|---|---|---|
| ① | `prepare_decode` builds this step's inputs as new small tensors, at a new address every step. `run_model` copies them into the top-left corner of the buffers, for example a 3×3 `block_tables` into the 512×16 buffer. Empty seats get `slot_mapping = -1` and `context_lens = 0` | [L204-L210](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L204-L210) |
| ② | `run_model` replays the graph for the smallest recorded batch size that fits. The replay passes no data | [L202](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L202), [L211](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L211) |
| ③ | Each node's kernel reads its inputs at the recorded addresses: the embedding reads `input_ids`, the rotary embedding reads `positions`, attention reads `block_tables` and `context_lens` | |
| ④ | `store_kvcache` writes this step's K and V into the slots listed in `slot_mapping`. Attention reads the history from the KV cache, block by block | [attention.py L61-L74](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L61-L74) |
| ⑤ | The last node copies the final hidden states into `outputs` | [L243](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L243) |
| ⑥ | `compute_logits` runs eagerly on `outputs[:bs]`, outside the graph | [L212](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L212) |

#190 stops at step ①: a table 17 entries wide cannot be copied into the 16-wide buffer, so the replay never starts.

### One graph, unpacked

Every node in the graph is the same three things: which kernel, its launch configuration, and its arguments, which are GPU memory addresses. Here is the graph for batch size 1, the one I dumped, with two nodes opened up:

```
CUDA graph: graphs[1]              (recorded at startup, kept by the GPU driver; one per batch size 1, 2, 4, 8, 16, …)
 │
 ├─ node #0  embedding
 ├─ node #1  rmsnorm
 ├─ …
 ├─ node #7  store_kvcache
 │    ├─ kernel ─────────► store_kvcache_kernel           (nano-vllm's Triton code)
 │    ├─ launch config     <<<1, 128, 0>>>
 │    └─ arguments
 │         ├─ slot_mapping ──► [GPU memory] graph buffer slot_mapping
 │         ├─ key, value ────► [GPU memory] graph memory pool: k, v from earlier nodes
 │         └─ k/v cache ─────► [GPU memory] KV cache
 │
 ├─ node #8  attention
 │    ├─ kernel ─────────► flash_fwd_splitkv_kernel       (FlashAttention's GPU code)
 │    ├─ launch config     <<<{1,4,8}, 128, 81920>>>
 │    └─ arguments
 │         ├─ block_table ───► [GPU memory] graph buffer block_tables, 16 entries per row  ← the width is recorded
 │         ├─ cache_seqlens ─► [GPU memory] graph buffer context_lens
 │         ├─ q ─────────────► [GPU memory] graph memory pool: q from earlier nodes
 │         └─ k/v cache ─────► [GPU memory] KV cache
 ├─ …  (28 layers, 14 nodes each)
 └─ node #394  MEMCPY ──────► [GPU memory] graph buffer outputs
```

- **kernel**: the code that runs on the GPU; the node only points to it
- **launch config**: how many blocks, and how many threads per block
- **arguments**: GPU memory addresses that the kernel reads and writes
- **`block_tables`** is just a block of GPU memory. Its only link to the graph is that node #8's arguments hold its address and its row width, 16

The `.dot` dump does not print the arguments, but they are recorded with each node.

### Every buffer the graph uses

| Buffer | Shape | Created by | Written each step by | Read inside the graph by |
|---|---|---|---|---|
| `input_ids` | 512 | `capture_cudagraph` | CPU copy (L204) | the embedding lookup |
| `positions` | 512 | `capture_cudagraph` | CPU copy (L205) | each layer's rotary embedding |
| `slot_mapping` | 512 | `capture_cudagraph` | CPU copy (L206-L207) | each layer's `store_kvcache` |
| `context_lens` | 512 | `capture_cudagraph` | CPU copy (L208-L209) | each layer's attention |
| `block_tables` | 512 × 16 | `capture_cudagraph` | CPU copy (L210) | each layer's attention |
| `outputs` | 512 × 1024 | `capture_cudagraph` | the graph's last node | `compute_logits`, on the CPU side of the step |
| KV cache | 28 layers × blocks × 256 tokens × 8 heads × 128 | `allocate_kv_cache`, before recording | each layer's `store_kvcache`, inside the graph | each layer's attention |
| graph memory pool | intermediate results | recording | nodes in the graph | the next nodes |

The CPU only writes the first five buffers and reads `outputs`. The KV cache and the memory pool are read and written only by kernels inside the graph.

### Addresses, block numbers and slots

"Buffer" here means only those five inputs and one output, not every tensor the model touches. Intermediate results such as the hidden states live in the graph's memory pool, and the weights never change. All of them sit at fixed addresses, but only the buffers are refilled by the CPU.

Three kinds of "where" are easy to confuse:

```
1. A node's arguments: GPU memory addresses, fixed when the graph is recorded
     "the block_tables buffer is at 0xB000", "the KV cache is at 0xA000"
2. The contents of block_tables: KV cache block numbers, data that changes every step
     "A's history is in KV blocks 5, 9 and 12"
3. The contents of slot_mapping: KV cache slot numbers, data that changes every step
     "A's new K and V go to slot 12×256+87"
```

`block_tables` is one of the buffers in the first sense, and it holds numbers of the second kind. The graph never knows the values. It records addresses, shapes and which kernel to launch, and the kernels read whatever is at those addresses when the graph is replayed.

So #190, in these terms: node #8 was recorded with "read `block_tables`, 16 entries per row". A sequence that needs a 17th block produces a table 17 entries wide, and the copy into the 16-wide buffer fails before the graph even runs. A wider tensor would sit at another address, which the recorded node never reads.

## Fix directions to weigh

1. Make the buffer wider from the start. How wide is enough?
2. When the table does not fit, run that step eagerly. PR #191 does this, and also clears the buffer.
3. Enforce `max_model_len`: stop or reject sequences at the limit, so the table never outgrows the buffer.

PRs #258 ("guard CUDA graph block table replay") and #263 ("validate request token limits") also mention this issue. I have not read them yet.

## Next

- Reproduce it: shrink `max_model_len` to 512, so the buffer has 2 columns. Then let a 500-token prompt generate past 512 tokens with `ignore_eos`, with CUDA graph on and off. Write down predictions first: how many tokens are generated before the error, which two numbers appear in the error message, and how long the sequence gets in eager mode.
- Compare the fix directions and the three PRs.

## Next questions

Harder questions to work through after the reproduction:

1. **Order at startup.** `ModelRunner.__init__` allocates the KV cache before recording the graphs. What would happen the other way round? Would anything raise? Hint: before allocation, `Attention.k_cache` is an empty tensor; see how `forward` handles that ([attention.py L57-L63](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L57-L63)).
2. **The issue's first claim.** Build a case where the stale columns in `block_tables` really produce a wrong result, or argue it cannot happen. Which condition does the argument rely on, and which code guarantees it?
3. **Cost of falling back to eager (PR #191).** In a batch of 16, one sequence reaches 4097 tokens. What happens on this step, the next, and the one after? When can the graph be used again? Compared with enforcing `max_model_len`, which is better for the user, and which for throughput?
4. **Cost of a wider buffer.** Size the buffer for 40960 tokens, the rotary limit. How big does `block_tables` get, and is memory the real cost? What else depends on the buffer's width, and could that slow down decode for short sequences? Does it fully fix #190? And does the issue's "one extra column" fix it, or move the crash from 4097 to 4353 tokens?
5. **What changes between two steps.** For A, B and C over two consecutive steps with no new block, which of the five input buffers change, and how? What if A grows from 768 to 769 tokens? How does this relate to [#175](https://github.com/GeeeekExplorer/nano-vllm/issues/175), rebuilding the inputs every step, and how could it be optimized?
6. **Predict the other graphs.** How many nodes do the graphs for batch sizes 2, 4 and 8 have, and around which batch size does each kernel switch? This one can be checked by dumping those graphs too.

## Questions for later: decode performance

These came up while reading this code. They are beyond #190:

- **Inputs are rebuilt on the CPU every step.** `prepare_decode` loops over every sequence in Python each step, even though most values change only slightly. Some rebuilding is unavoidable, because the batch changes every step as sequences finish, join or get preempted (continuous batching). This is upstream [#175](https://github.com/GeeeekExplorer/nano-vllm/issues/175), with PRs [#176](https://github.com/GeeeekExplorer/nano-vllm/pull/176) and [#253](https://github.com/GeeeekExplorer/nano-vllm/pull/253). Two ideas to read about: keeping input buffers alive and updating only what changed, and SGLang's overlap scheduling, which prepares the next batch on the CPU while the GPU runs the current one.
- **One very long sequence among short ones.** Padding is not the problem; load imbalance is. The long sequence's attention keeps working after the short ones finish. Split-KV ("Flash-Decoding") cuts a long history into chunks computed in parallel. FlashAttention's `flash_attn_with_kvcache` has a `num_splits` argument, default 0, which picks the split count automatically. The recorded graphs show it at work: attention is split in 4 at batch size 1 and not split at batch size 16.
- **Batch padding to recorded sizes.** 17 sequences run as 32. More recorded sizes would waste fewer rows, but cost more startup time and memory.
