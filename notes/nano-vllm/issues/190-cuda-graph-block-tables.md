# [nano-vllm #190] CUDA graph replay fails once a sequence grows past max_model_len

- Upstream issue: https://github.com/GeeeekExplorer/nano-vllm/issues/190 · Same bug, reported earlier: [#106](https://github.com/GeeeekExplorer/nano-vllm/issues/106) · Open PRs: [#270](https://github.com/GeeeekExplorer/nano-vllm/pull/270) (fixes the root cause; it did not link this issue, so I [commented there](https://github.com/GeeeekExplorer/nano-vllm/pull/270#issuecomment-6064191717) with the reproduction, and its author then added `Fixes #190`) and [#191](https://github.com/GeeeekExplorer/nano-vllm/pull/191) (avoids the crash) · Closed without merging: [#258](https://github.com/GeeeekExplorer/nano-vllm/pull/258), [#263](https://github.com/GeeeekExplorer/nano-vllm/pull/263)
- Status: studied. Reproduced on unmodified upstream, found why nothing catches it earlier, and tested the two open PRs with the same script. No PR from me: #270 already fixes the root cause, so I posted the reproduction on it instead (2026-10-08). Its author then added `Fixes #190` to the PR (2026-10-09).
- Commit read: `bb823b3`, unmodified upstream; the PRs at their heads, #270 `b8996b2` and #191 `b386ae2`
- Written: 2026-10-03 · Updated: 2026-10-09
- Study Q&A (Chinese), with every question I asked along the way: [QA-cuda-graph.md](../QA-cuda-graph.md)
- Side quest: while I was recording the graphs, `torch.compile` failed in the warmup before capture with `PermissionError: [WinError 5]`. That is a PyTorch 2.6 bug on Windows, unrelated to #190. I traced it and documented it in triton-windows's README: [PR #56](https://github.com/triton-lang/triton-windows/pull/56), [write-up](../../triton-windows/issues/56-pytorch-2.6-os-replace.md)

<!-- Also add a row to CONTRIBUTIONS.md and to the Contributions table in README.md, and keep its status in sync with this page. -->

## TL;DR

- **The bug:** with CUDA graph on (`enforce_eager=False`), decode crashes with `RuntimeError: The expanded size of the tensor (16) must match the existing size (17)` as soon as any sequence in the batch grows past `max_model_len` (4096 by default).
- **Why:** CUDA graph replays GPU work recorded at startup, and that work reads fixed buffers. The `block_tables` buffer has `max_model_len / 256` = 16 columns, so it fits sequences of up to 4096 tokens. Nothing stops a sequence from growing longer: neither the scheduler nor `SamplingParams` checks `max_model_len`. A 4097-token sequence needs a 17th column, and the copy into the buffer fails.
- **Only the CUDA graph path is affected.** Prefill, `enforce_eager=True`, and batches of more than 512 sequences build their tensors fresh every step. `example.py` sets `enforce_eager=True`, so it never hits this.
- **Reproduced** on unmodified upstream. The fix the issue suggests, one extra buffer column, only moves the crash by one block ([Reproduction](#reproduction)). Nothing checks the table's width before the copy; the limit the buffer was sized for is assumed but never enforced ([Why nothing catches it earlier](#why-nothing-catches-it-earlier)).
- **The open PRs, tested on the same script:** [#270](https://github.com/GeeeekExplorer/nano-vllm/pull/270) fixes the root cause by rejecting any request whose prompt plus `max_tokens` exceeds `max_model_len`, but one bad request fails the whole call. [#191](https://github.com/GeeeekExplorer/nano-vllm/pull/191) runs a step eagerly when the table would not fit: no crash, but the sequence grows past `max_model_len`. Details: [Fixes compared](#fixes-compared-the-open-prs-tested). What I take from it: [Thoughts](#thoughts).
- **Posted on #270** (2026-10-08): #270 never mentioned #190, so I [commented](https://github.com/GeeeekExplorer/nano-vllm/pull/270#issuecomment-6064191717) with this reproduction on both versions. GitHub now shows the link on #190's page too. The next day its author added `Fixes #190` to the PR description, so #190 will be closed when #270 is merged.

## What I learned

- What cuda_graph looks like? -> print the results and checked
- Difference on batch size -> the kernel changes w.r.t. batch size. for batch = 1 or 16, the mat_mul change from gemv to cutlass, attention changed from 4 pieces to more pieces, 
- How long sequence is split -> flash_fwd_splitkv + combine
- Understanding of block table, cuda graph, kernel -> cuda graph records a series of kernel actions, and at each action, kernels fetches data based on block table info. 
- The issue's suggested fix (add one extra column) does not fix it  -> This is not the root cause, as long as we did not check the length, there will be error.  
- Three steps: a block is added, the table is rebuilt, then copied into the graph buffer -> the first two do not check size, leads to copy error in third step. 
- Seems like nano-vllm has many default size limits that are assumed but have not been enforced ( #274, #279, #190) -> Not a problem for simplicity, but good for learning. 
- Fix in #270 -> it checks prompt + max_tokens <= max_model_len, so stop the request before the add_request, solve this issue from the beginning, but will block the whole batch. -> Give a comment to link the issue after reproducing; also analyzed the pros and cons in notes.
- Fix in #191 -> use the eager mode, did not change the logic, and model will ultimately break when reaching the limit of RoPE! | also, the second condition is not useful as it will always be skipped. 
- In #191, need to be more carefully on CPU and GPU synchronization.  
- Compared multiple PR and their solution. 
- My insights: a good pr should not just work, but also need to design from a higher level, think of the root cause, the cost of certain operations, etc. 

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

### Why nothing catches it earlier

Within one decode step, three pieces of code touch the table, and each one assumes something else keeps it in bounds:

| Step | Code | What it does | Why it does not stop the overflow |
|---|---|---|---|
| 1 | [`may_append`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L106-L108) | adds a block when the length crosses a multiple of 256 | it only checks for a free KV block; it does not know the graph buffer's width, which belongs to `ModelRunner` |
| 2 | [`prepare_block_tables`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L123-L127) | builds this step's table, padded to the longest row | it builds whatever width the lists need |
| 3 | [`run_model` L210](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L210) | copies the table into the buffer | slicing past the end of a tensor clips silently (`buf[:1, :3]` on a 2-wide buffer is 1×2), so only the copy itself fails |

All three happen in the same step, milliseconds apart. The buffer's width comes from `max_model_len`, on the assumption that no sequence grows past it, but nothing enforces that. A check could live before step 1: finish the sequence at `max_model_len`, as vLLM does with `finish_reason="length"`, so the table never outgrows the buffer. Or before step 3: fall back to eager, as PR #191 does, which avoids the crash but still lets the sequence grow past `max_model_len`. Both are tested under [Fixes compared](#fixes-compared-the-open-prs-tested).

The crash is the lucky outcome. If the copy also clipped silently, attention would need a third block number but find only two per row, and would read into the next seat's row: it would compute with another sequence's KV cache, without any error.

### Two claims in the issue to check

- "The buffer is not cleared before copying." Columns beyond this step's width keep values from earlier steps. Does attention ever read them, given that it stops at `context_lens`?
- "Allocate one extra column." Does that fix the bug, or only move the crash from 4097 tokens to 4353? **It only moves it**, as measured under [Reproduction](#reproduction).

## Reproduction

`max_model_len=512`, so the buffer has 2 columns (512 tokens). One 500-token prompt with `ignore_eos` and `max_tokens=500`, so the sequence would grow to 1000 tokens. CUDA graph on. I ran it on unmodified `bb823b3`, then on a copy with the issue's fix, one extra column:

| Version | Buffer | Holds up to | Result |
|---|---|---|---|
| upstream `bb823b3` | 2 columns | 512 tokens | `RuntimeError: The expanded size of the tensor (2) must match the existing size (3) … Target sizes: [1, 2]. Tensor sizes: [3]` |
| one extra column | 3 columns | 768 tokens | `RuntimeError: … (3) must match the existing size (4) … Target sizes: [1, 3]. Tensor sizes: [4]` |

Both fail the moment the table needs one more column than the buffer has. The extra column moves the crash by one block, from 513 tokens to 769, and any fixed width is outgrown as long as nothing stops the sequence. The test had to push past the new boundary: with `max_tokens=100` the sequence stops at 600 tokens, never reaches 768, and the same run would have suggested that the fix works.

In the message, `Target sizes` is the slice of the buffer, already clipped to the buffer's width, and `Tensor sizes: [3]` is this step's 1×3 table: PyTorch drops leading dimensions of size 1 from the source before an indexed copy.

## Fixes compared: the open PRs, tested

| PR | Approach | Status (2026-10-09) | Fixes #190? |
|---|---|---|---|
| [#270](https://github.com/GeeeekExplorer/nano-vllm/pull/270) | Reject a request in `add_request` when prompt + `max_tokens` > `max_model_len` (`ValueError`); adds 10 CPU-only tests | open, mergeable, no review yet; my comment is the first, and the description now says `Fixes #190` | **Yes, at the root**: no sequence can outgrow the buffer |
| [#191](https://github.com/GeeeekExplorer/nano-vllm/pull/191) | Before each decode step, run eagerly if the table would not fit; one extra buffer column; clear the buffer before each copy | open, conflicts with main | Avoids the crash, but sequences still grow past `max_model_len` |
| [#258](https://github.com/GeeeekExplorer/nano-vllm/pull/258) | Same idea as #191 | closed without merging | — |
| [#263](https://github.com/GeeeekExplorer/nano-vllm/pull/263) | First version of #270, closed by its author the day #270 was opened | closed without merging | — |

#270 did not mention #190, so searching the PRs for "190" did not find it; I found it by searching for `max_model_len`. A maintainer reading #190 would not have known about it either. So on 2026-10-08 I [commented on #270](https://github.com/GeeeekExplorer/nano-vllm/pull/270#issuecomment-6064191717): the reproduction below, run on `main` and on the PR, and a suggestion to add `Fixes #190`. Before posting, I ran the exact script from the comment on both versions. The author added `Fixes #190` the next day, without replying.

### The same script on four versions

Same setup as [Reproduction](#reproduction): `max_model_len=512`, a 500-token prompt, `max_tokens=500`, `ignore_eos`, CUDA graph on. Each PR was fetched from `pull/<N>/head` and run from its own worktree.

| Version | Buffer | Result |
|---|---|---|
| upstream `bb823b3` | 2 columns | crashes at 513 tokens: `(2) must match … (3)` |
| one extra column | 3 columns | crashes at 769 tokens: `(3) must match … (4)` |
| PR #270 | 2 columns | rejected before the first step, no tokens generated: `ValueError: request requires 1000 tokens, exceeding max_model_len=512` |
| PR #191 | 3 columns | no crash: 500 tokens generated, so the sequence reached 1000 tokens, past `max_model_len=512` |

### PR #270: reject the request when it comes in

The fix is one line in `add_request` ([llm_engine.py L47](https://github.com/GeeeekExplorer/nano-vllm/blob/b8996b26a70727e73f8747f976303ae87d93a94e/nanovllm/engine/llm_engine.py#L47)), plus the check in [`SamplingParams.validate_request_length`](https://github.com/GeeeekExplorer/nano-vllm/blob/b8996b26a70727e73f8747f976303ae87d93a94e/nanovllm/sampling_params.py#L19-L27):

```python
sampling_params.validate_request_length(len(prompt), self.max_model_len)   # new
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

Why it fixes #190: a sequence is at most prompt + `max_tokens` ≤ `max_model_len` tokens long, so it needs at most `ceil(max_model_len / 256)` blocks, which is exactly the buffer's width.

Problems I see in the code (the second and third tested on 2026-10-08, below):

1. **One bad request fails the whole call.** `generate()` raises, so the valid requests in the same call get nothing. This is the same weakness I found in PR #277's fix for #274.
2. **Requests added before the bad one stay in the engine.** `generate()` adds requests one at a time ([L72](https://github.com/GeeeekExplorer/nano-vllm/blob/b8996b26a70727e73f8747f976303ae87d93a94e/nanovllm/engine/llm_engine.py#L72)), so the ones before the bad request are already queued when it raises. The next `generate()` runs them too, and returns their results along with its own, because it collects every finished sequence by ID ([L87-L90](https://github.com/GeeeekExplorer/nano-vllm/blob/b8996b26a70727e73f8747f976303ae87d93a94e/nanovllm/engine/llm_engine.py#L87-L90)).
3. **Strict.** It rejects on the worst case, even if the model would stop at EOS long before. vLLM's OpenAI-compatible server rejects such requests too, so this is a trade-off rather than a bug. It is also one token stricter than the crash, which I think is right.

#### The boundary: why 13 tokens pass on `main` and 14 crash

A 500-token prompt with `max_model_len=512`, on both versions:

| `max_tokens` | Longest input to a decode step | Blocks | `main` | #270 |
|---|---|---|---|---|
| 12 | 511 tokens | 2 | 12 tokens generated, sequence 512 | same |
| 13 | 512 tokens | 2 | 13 tokens generated, sequence 513: no crash, but one past `max_model_len` | `ValueError: request requires 513 tokens` |
| 14 | 513 tokens | 3 | crash: `(2) must match … (3)` | rejected (514 > 512; not run) |

The last generated token is never fed back into the model. A decode step takes a sequence of length L, writes the KV of its last token, and appends one new token. When the count reaches `max_tokens`, the request finishes ([scheduler.py L89](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L89)), and the new token is never processed. So a request only needs KV slots for prompt + `max_tokens` − 1 tokens, and the crash starts one token later than #270's limit. It happens on the step whose input is 513 tokens long: 513 % 256 == 1, so `may_append` adds a third block ([L69](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L69)), and the copy at [L210](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L210) fails. In the original reproduction (`max_tokens=500`), that is after 13 tokens.

So #270 enforces what `max_model_len` means, the longest a sequence may get, not the crash point. `main` with 13 tokens is a "no crash, still wrong" case: the sequence ends at 513.

#### Leftover requests

One `LLM` object, two calls, in eager mode since this has nothing to do with CUDA graph. A and B have the same 10-token prompt, with `max_tokens` 5 and 7 and `ignore_eos`, so the token count tells whose result is whose:

| First call | Queue after the `ValueError` | Second call, `generate([B])`, returns |
|---|---|---|
| `generate([A, too_long])` | A still waiting | 2 results: `outputs[0]` is A's (5 tokens), `outputs[1]` is B's (7 tokens) |
| `generate([too_long, A])` | empty | 1 result, B's |

A caller that catches the `ValueError` and goes on, say a script that skips a bad batch, gets someone else's result at `outputs[0]`, with no error. Whether anything leaks depends on where the bad request sits in the list. (`seq_id` keeps counting across calls, and warmup uses 0 to 15, so A was 16 and sorted before B.)

On `main`, in eager mode, the same first call raises nothing: the too-long request runs to the end, 1000 tokens, past `max_model_len`.

#### How I would fix 1 and 2

Don't raise; finish only the bad request, as my PR #280 does for requests that cannot fit in the KV cache. One detail matters: the skipped request still needs an entry in the outputs, an empty result with a `finish_reason`. If it just disappears, every later result shifts by one, the same silent mismatch as the leftover requests. #280 does this: `Scheduler.add` marks the request finished with `finish_reason="prompt_too_long"` instead of queuing it ([scheduler.py L23-L28](https://github.com/taigaoma1997/nano-vllm/blob/7a75224fede967380a1d7c7311b86e331dc21116/nanovllm/engine/scheduler.py#L23-L28)), and `generate()` records it right away under its `seq_id` ([llm_engine.py L71-L75](https://github.com/taigaoma1997/nano-vllm/blob/7a75224fede967380a1d7c7311b86e331dc21116/nanovllm/engine/llm_engine.py#L71-L75)).

A smaller change that keeps #270's `ValueError`: tokenize and check every prompt in `generate()` before adding any. Nothing is left behind, but one bad request still fails the whole call.

### PR #191: run the step eagerly when the table does not fit

Running it today took one detour, unrelated to the fix. The PR is based on `2f21442` (2025-11-04). With transformers 5.16, the model config's `rope_scaling` comes back as a dict (`{'rope_theta': 1000000, 'rope_type': 'default'}`), and the old `get_rope`, cached with `@lru_cache`, cannot hash it: `TypeError: unhashable type: 'dict'`. Upstream fixed that in [`8d63a98`](https://github.com/GeeeekExplorer/nano-vllm/commit/8d63a98c03805e54e9a422fd83fff7a4780c17dc) (2026-04-14), which also rewrote much of `model_runner.py`; that is probably why #191 now conflicts. #191 does not touch the two rope files, so I took them from `8d63a98` and kept the rest of the PR as it is.

The fix is a check before each decode step, in [`can_use_decode_graph`](https://github.com/GeeeekExplorer/nano-vllm/blob/b386ae2f912c943dd1406bf068fced8e302c622f/nanovllm/engine/model_runner.py#L189-L199):

```python
if int(context.context_lens.max().item()) > self.config.max_seq_len_to_capture:   # ①
    return False
if context.block_tables.size(1) > self.graph_vars["block_tables"].size(1):        # ②
    return False
```

`max_seq_len_to_capture` is a new setting that defaults to `max_model_len`; older vLLM versions had a setting with the same name and meaning. The buffer becomes `ceil(max_seq_len_to_capture / 256) + 1` columns wide ([L234](https://github.com/GeeeekExplorer/nano-vllm/blob/b386ae2f912c943dd1406bf068fced8e302c622f/nanovllm/engine/model_runner.py#L234)), 3 in my run.

- **① always fires first, so ② never decides.** If every sequence is at most 512 tokens long, each needs at most `ceil(512 / 256) = 2` blocks, so the table is at most 2 columns wide, narrower than the 3-column buffer. In my run, the eager steps started at 513 tokens.
- **The extra column is never used.** On graph steps each sequence has at most 2 blocks, so the third column only ever holds the `-1` from `fill_(-1)`.
- **② is the check that matters, and it is free.** It is exactly the condition under which the copy fails, so it alone prevents the crash. A tensor's shape is known on the CPU, so it costs nothing to read.
- **① costs a CPU–GPU sync on every step**, including the steps that use the graph. `context_lens` is on the GPU, so `.item()` makes the CPU wait until the GPU has finished the queued copies and the `max`, and then copies the result back. Meanwhile the GPU sits idle until the CPU queues the replay. In nano-vllm the cost is small, because each step already syncs once when it reads the sampled tokens ([L225](https://github.com/GeeeekExplorer/nano-vllm/blob/b386ae2f912c943dd1406bf068fced8e302c622f/nanovllm/engine/model_runner.py#L225)). In an engine that prepares the next step on the CPU while the GPU runs the current one, such as SGLang with its overlap scheduler, it would cancel that overlap. And the value is already on the CPU: `prepare_decode` builds `context_lens` from `len(seq)` in a Python list ([L172](https://github.com/GeeeekExplorer/nano-vllm/blob/b386ae2f912c943dd1406bf068fced8e302c622f/nanovllm/engine/model_runner.py#L172)).
- **No crash does not mean correct.** The sequence went past `max_model_len`, and the output was still fine only because I had set 512 low on purpose. The hard limit is the rotary embedding table, 40960 positions for Qwen3-0.6B ([rotary_embedding.py L44](https://github.com/GeeeekExplorer/nano-vllm/blob/b386ae2f912c943dd1406bf068fced8e302c622f/nanovllm/layers/rotary_embedding.py#L44)). With `max_model_len` set to 40960, #191 would let a sequence index past that table. It moves the failure from `max_model_len` to `max_position_embeddings`.

### Side by side

| Fix | Where it checks | Crash? | Can a sequence pass `max_model_len`? | Who pays for one bad request | Extra cost per step |
|---|---|---|---|---|---|
| One extra column (the issue's suggestion) | nowhere | yes, one block later | yes | the whole batch (crash) | none |
| PR #191 | before each decode step | no | yes, up to the rotary limit | open: [next question 3](#next-questions) | a CPU–GPU sync; eager steps past the limit |
| PR #270 | when a request comes in | no | no | the whole `generate()` call, plus leftover queued requests | none |
| Two checks (not written) | when a request comes in (reject only that one) and after each step (finish at `max_model_len` with `finish_reason="length"`) | no | no | only that request | none: one integer comparison on the CPU |

The last row is how I would fix it, in the same style as my PR #280 for the KV cache limit, with the rejected request still getting its place in the outputs ([above](#how-i-would-fix-1-and-2)). #270 already fixes the root cause, so I have not written it.

## Thoughts

What this issue taught me beyond the bug itself:

1. **Enforce a limit where requests come in and grow, not where it finally breaks.** The crash happens in `ModelRunner`, at L210, but the limit is about the request: whether it may come in, and when it must stop. Fixes at the crash site, a wider buffer or an eager fallback, leave the limit unenforced and move the failure somewhere else. Two choke points cover it: `Scheduler.add` when a request comes in, and `Scheduler.postprocess` after each new token. These are the same two places my PR #280 uses for the KV cache limit.
2. **Every dimension of a CUDA graph buffer needs a guard.** A recorded graph has fixed shapes. nano-vllm guards the batch dimension: more than 512 sequences run eagerly. It does not guard the table's width. When I read another engine's CUDA graph code, I can list each buffer dimension and look for what keeps it in bounds.
3. **Decide who pays for one bad request.** Crashing, raising from `generate()` as #270 does, and finishing only that request are three different answers. Only the last keeps the other requests' results. The same question came up with #274 and PR #277.
4. **A check on the hot path should read data that is already on the CPU.** Shapes and Python lists are free; `.item()` on a GPU tensor is a sync. When two checks overlap, keep the one that matches the failure exactly, ② in #191, and drop the other.
5. **Test a fix past its own new boundary, and do not take "no crash" as proof.** The extra column passes any test that stops before 768 tokens. #191 runs my reproduction cleanly, but only because the limit it ignores was set low; the next limit is the rotary table.
6. **Search for existing fixes by mechanism, not only by issue number.** #270 did not mention #190 until I pointed it out. And an old PR can fail today for reasons unrelated to its fix, as #191 did after the transformers change, so I separate "does the fix work" from "does the PR's base still run".

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

## Next

- Measure the cost of the eager fallback itself: generate the same number of tokens once all on the graph and once mostly eager.
- Watch my comment on #270 for replies.

## Next questions

Harder questions to work through after the reproduction:

1. **Order at startup.** `ModelRunner.__init__` allocates the KV cache before recording the graphs. What would happen the other way round? Would anything raise? Hint: before allocation, `Attention.k_cache` is an empty tensor; see how `forward` handles that ([attention.py L57-L63](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L57-L63)).
2. **The issue's first claim.** Build a case where the stale columns in `block_tables` really produce a wrong result, or argue it cannot happen. Which condition does the argument rely on, and which code guarantees it?
3. **Cost of falling back to eager (PR #191).** In a batch of 16, one sequence reaches 4097 tokens. What happens on this step, the next, and the one after? When can the graph be used again? Compared with enforcing `max_model_len`, which is better for the user, and which for throughput? Hint: #191's check is under [Fixes compared](#pr-191-run-the-step-eagerly-when-the-table-does-not-fit); what does ① compare across the batch?
4. **Cost of a wider buffer.** Size the buffer for 40960 tokens, the rotary limit. How big does `block_tables` get, and is memory the real cost? What else depends on the buffer's width, and could that slow down decode for short sequences? Does it fully fix #190? (The issue's "one extra column" does not: it only moves the crash, see [Reproduction](#reproduction).)
5. **What changes between two steps.** For A, B and C over two consecutive steps with no new block, which of the five input buffers change, and how? What if A grows from 768 to 769 tokens? How does this relate to [#175](https://github.com/GeeeekExplorer/nano-vllm/issues/175), rebuilding the inputs every step, and how could it be optimized?
6. **Predict the other graphs.** How many nodes do the graphs for batch sizes 2, 4 and 8 have, and around which batch size does each kernel switch? This one can be checked by dumping those graphs too.

## Questions for later: decode performance

These came up while reading this code. They are beyond #190:

- **Inputs are rebuilt on the CPU every step.** `prepare_decode` loops over every sequence in Python each step, even though most values change only slightly. Some rebuilding is unavoidable, because the batch changes every step as sequences finish, join or get preempted (continuous batching). This is upstream [#175](https://github.com/GeeeekExplorer/nano-vllm/issues/175), with PRs [#176](https://github.com/GeeeekExplorer/nano-vllm/pull/176) and [#253](https://github.com/GeeeekExplorer/nano-vllm/pull/253). Two ideas to read about: keeping input buffers alive and updating only what changed, and SGLang's overlap scheduling, which prepares the next batch on the CPU while the GPU runs the current one.
- **One very long sequence among short ones.** Padding is not the problem; load imbalance is. The long sequence's attention keeps working after the short ones finish. Split-KV ("Flash-Decoding") cuts a long history into chunks computed in parallel. FlashAttention's `flash_attn_with_kvcache` has a `num_splits` argument, default 0, which picks the split count automatically. The recorded graphs show it at work: attention is split in 4 at batch size 1 and not split at batch size 16.
- **Batch padding to recorded sizes.** 17 sequences run as 32. More recorded sizes would waste fewer rows, but cost more startup time and memory.
