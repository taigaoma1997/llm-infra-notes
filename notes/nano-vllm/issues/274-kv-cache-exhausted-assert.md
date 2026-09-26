# [nano-vllm #274] Engine crashes on assert scheduled_seqs when one sequence outgrows the KV cache

- Upstream issue: https://github.com/GeeeekExplorer/nano-vllm/issues/274
- Status: investigating
- Commit tested: `bb823b3` plus my local learning patches (tracing and a KV block cap, see Reproduce)
- Written: 2026-09-25

## Environment

```
date:     2026-09-25
os:       Windows 11 (MINGW64_NT-10.0-26200)
gpu:      NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 566.24
python:   3.10.21
torch:    2.6.0+cu124 (CUDA 12.4)
triton:   3.2.0 (triton-windows, added by hand: collect_env.sh looks for the package name "triton")
flash-attn: 2.8.3
transformers: 5.16.1
nano-vllm: 0.2.0
repo:     nano-vllm @ bb823b3 (uncommitted changes)
model:    Qwen3-0.6B
```

On Windows, nano-vllm also needs the `gloo` backend instead of NCCL and `USE_LIBUV=0` to start (upstream [#261](https://github.com/GeeeekExplorer/nano-vllm/issues/261)).

## Symptom

When a single sequence needs more KV cache blocks than the whole cache has, and there is no other running sequence to preempt, `Scheduler.schedule()` ends a decode step with an empty batch and stops on a bare `assert scheduled_seqs`. The process dies with `AssertionError` and no hint about the cause:

```
File "nanovllm/engine/llm_engine.py", line 74, in step
    seqs, is_prefill = self.scheduler.schedule()
File "nanovllm/engine/scheduler.py", line 100, in schedule
    assert scheduled_seqs
AssertionError
```

(Line 100 is in my patched copy. In upstream `bb823b3` it is [scheduler.py#L71](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L71).)

With a normal KV cache this is hard to hit: on an 8 GB GPU the cache holds about 163 blocks, or about 42K tokens, far more than the default `max_model_len` of 4096. To trigger it, I shrink the cache to 2 blocks.

## Reproduce

`repro_274.py`, run against my local copy of nano-vllm. It uses two options I added for learning: `kvcache_blocks_limit` (cap the number of KV blocks) and `trace_file` (write every scheduling step to a log). These live in my local checkout for now and will go to a `learning` branch on my fork later.

```python
import os
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

def main():
    # The whole KV cache is 2 blocks = 512 tokens.
    llm = LLM(MODEL, enforce_eager=True, kvcache_blocks_limit=2,
              trace_file="repro_274.log", trace_detail_steps=20)
    # A 500-token prompt fills both blocks. With ignore_eos the sequence keeps growing,
    # and at 513 tokens it needs a third block.
    llm.generate([[100] * 500], SamplingParams(max_tokens=100, ignore_eos=True))

if __name__ == "__main__":
    main()
```

```powershell
conda activate learn-vllm
$env:USE_LIBUV = "0"
python repro_274.py
```

The upstream issue also has a pure-Python repro that drives `Scheduler` and `BlockManager` directly, without a GPU or model weights.

### What the trace shows

Step 1 prefills 500 tokens into both blocks and samples token 501. Steps 2 to 13 decode one token each. The last good step and the crash:

```
==================== STEP 13 | DECODE | 1 seqs ====================
[state] before: waiting=[] running=[4] free_blocks=0/2
[sched] DECODE 1 seqs, 1 new token each: seq4(len 512)
[prep ] seq4: input 284(' =') pos=511 ctx_len=512 -> slot 511 = block 1*256+255 block_table=[0, 1]
[block] seq4 logical block #1 (phys 1) is now FULL -> hash 83d50611 (chained with previous block's hash) registered for prefix reuse
[post ] seq4 += 128253(' những') -> len 513 (completion 13/100)
[time ] 1 seqs, 1 tok, 65.8 ms, 15 tok/s | after: waiting=[] running=[4] free_blocks=0/2

==================== STEP 14 | CRASHED ====================
[state] before: waiting=[] running=[4] free_blocks=0/2
[sched] seq4 needs a new block, none free, nobody left to evict -> PREEMPT itself
[sched] PREEMPT seq4 (len 513): release its 2 blocks, back to FRONT of waiting; must re-prefill 513 tok later (prefix cache may save some)
[block] seq4 dealloc: 2 blocks back to free pool [1, 0]; hashes kept, so content stays reusable until the block is recycled
[crash] exception raised during this step, see the traceback in the console
```

At step 13 the sequence writes its 512th token into the last slot of block 1 (slot 511), and the sampled token makes it 513 tokens long. At step 14 that 513th token has nowhere to go.

## Root cause

The decode half of `schedule()` ([scheduler.py#L57-L73](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L57-L73)):

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())   # evict the youngest other sequence
        else:
            self.preempt(seq)                  # nobody else left: evict itself
            break                              # skips the else below, so seq is not scheduled
    else:
        ...
        scheduled_seqs.append(seq)
assert scheduled_seqs
```

1. The decode loop assumes at least one sequence can always run. Normally that holds: when blocks run out, the scheduler preempts the youngest running sequence and uses its blocks.
2. The assumption breaks when the only running sequence needs a new block. `can_append` fails ([block_manager.py#L103-L104](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L103-L104)), the sequence preempts itself and `break`s, the batch is empty, and the assert fires.
3. Removing the assert does not help. The preempted sequence goes back to `waiting` with 513 tokens, which need 3 blocks. The cache has 2 in total, so `can_allocate` returns -1 on every later step ([block_manager.py#L58-L73](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L58-L73)). The engine would loop forever on empty batches: the request can never finish, so this is not a temporary stall.

The underlying gap: nothing checks that a request fits in the KV cache at all. nano-vllm does not enforce `max_model_len` on generation length, and it does not check at startup that the cache can hold at least one `max_model_len` sequence.

## Fix

Not written yet. The options I am weighing:

- Raise a clear error in `schedule()` instead of the bare assert. This is what upstream PR [#277](https://github.com/GeeeekExplorer/nano-vllm/pull/277) does. Its author notes that the issue's suggestion, to treat the empty batch as a stall and retry, would hang forever.
- Reject the request up front in `add_request` if prompt length plus `max_tokens` exceeds the cache capacity.
- Check at startup that `num_kvcache_blocks * block_size >= max_model_len`, and enforce `max_model_len` during generation. As far as I know, vLLM does both.

## Verification

To do once I have a fix:

- `repro_274.py` should fail fast with a readable error, or reject the request, instead of `AssertionError`.
- `trace_example.py`, where preemption does work (there is a younger sequence to evict), should still finish all four requests with `preemptions: 1`.

## What I learned

- Preemption only works when there is someone else to evict. The self-preemption branch exists, but nothing handles what comes after it.
- An empty batch here means the request can never be served, not that the engine should wait. The issue and the PR disagree on this, and tracing the code shows the PR is right.
- `max_model_len` in nano-vllm only sizes buffers (warmup, CUDA Graph block tables). It is not an enforced limit, which is also behind upstream #190.
