# [nano-vllm #274] Engine crashes on assert scheduled_seqs when one sequence outgrows the KV cache

- Upstream issue: https://github.com/GeeeekExplorer/nano-vllm/issues/274
- Status: root cause found; fix written and tested locally, not submitted upstream
- Commit tested: `bb823b3` plus my local learning patches (tracing and a KV block cap, see Reproduce)
- Written: 2026-09-25 · Updated: 2026-09-28 (scheduler-only simulation, a second path to the crash, a test of PR #277, and my own fix)

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

## Digging further with a scheduler-only simulation

Running the full engine takes a minute or two per try, mostly warmup. To test many scenarios quickly, I wrote a small script that drives `Scheduler` and `BlockManager` directly, with no GPU and no model weights. Each "step" calls `schedule()` and then `postprocess()` with a dummy token, which is all the scheduler sees of the model anyway. Requests use `ignore_eos=True`, so each one runs to `max_tokens`.

<details>
<summary><code>sched_sim.py</code></summary>

```python
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def run(title, num_blocks, requests):
    print(f"\n=== {title} ({num_blocks} blocks = {num_blocks * 256} tokens) ===")
    cfg = SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=16384, eos=-1,
                          kvcache_block_size=256, num_kvcache_blocks=num_blocks)
    sch, names = Scheduler(cfg), {}

    def report(step, seq):
        # upstream has no finish_reason; shown as "-"
        print(f"  step {step:>2}: {names[seq.seq_id]} finished, {getattr(seq, 'finish_reason', '-')}, "
              f"{seq.num_completion_tokens} tokens")

    for name, prompt_len, max_tokens in requests:
        seq = Sequence([100] * prompt_len, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
        names[seq.seq_id] = name
        sch.add(seq)
        if seq.is_finished:  # rejected on arrival
            report(0, seq)
    for step in range(1, 2000):
        if sch.is_finished():
            print(f"  all finished after {step - 1} steps")
            return
        try:
            seqs, is_prefill = sch.schedule()
        except Exception as e:
            print(f"  step {step:>2}: {type(e).__name__}: {e}")
            print(f"           running={[names[s.seq_id] for s in sch.running]} "
                  f"waiting={[names[s.seq_id] for s in sch.waiting]}")
            return
        if is_prefill:
            print(f"  step {step:>2}: prefill {[names[s.seq_id] for s in seqs]}")
        # the model only has to produce one token per sequence; a dummy token 1 stands in
        sch.postprocess(seqs, [1] * len(seqs), is_prefill)
        for seq in seqs:
            if seq.is_finished:
                report(step, seq)


if __name__ == "__main__":
    # (name, prompt length, max_tokens)
    run("B first, then D", 4, [("B", 1000, 100), ("D", 10, 20)])
    run("D first, then B", 4, [("D", 10, 20), ("B", 1000, 100)])
    run("prompt too long", 2, [("P", 600, 10)])
    run("prompt too long, with others", 4, [("A", 10, 20), ("P", 1100, 10), ("C", 10, 5)])
```

</details>

It runs against whichever `nanovllm` is on the Python path. The script lives next to my local nano-vllm checkout as `sched_sim.py`. To run it on unpatched upstream code:

```bash
git -C nano-vllm archive bb823b3 | tar -x -C upstream     # clean copy, no local patches
PYTHONPATH=upstream python sched_sim.py
```

Before I wrote the fix, my local copy (which then only added logging) and a clean `bb823b3` gave identical output. The output blocks in this section come from that unfixed code, with my annotations of the state added between the printed lines. [Verification](#verification) compares before and after the fix.

### 1. Self-preemption on its own is correct

The `preempt(seq); break` branch is not the bug by itself. A dry run with two requests and a full 4-block cache:

| | Length | Blocks | Needs a new block this step? |
|---|---|---|---|
| A (older) | 300 | `[0, 1]` | no (300 % 256 = 44) |
| B (younger) | 513 | `[2, 3]` | yes (513 % 256 = 1) |

The decode loop takes A first. A needs nothing, so it is scheduled. Then it takes B. `running` is now empty and B needs a block, so B preempts itself. The step still runs A. On the next steps B cannot be admitted: it needs 3 blocks and only 2 are free. Once A finishes, 4 blocks are free and B comes back. Its two old blocks still hold their hashes and contents, so it hits the prefix cache and recomputes only one token.

The crash needs two things together: the batch ends up empty, and the evicted request can never fit again. Whether a request can come back depends on the cache's *total* number of blocks, not on how many are free right now.

### 2. A request outgrows the cache only when it is the only one running

I first assumed a second request D could still be running at the moment B outgrows the cache, so the crash would come later, after D finished. That cannot happen. To need block T+1 of a T-block cache, B must already hold all T blocks, and every running request holds at least one block. So when B outgrows the cache, nothing else is running. The simulation agrees:

```
=== B first, then D (4 blocks = 1024 tokens) ===
step 1:  prefill B -> B holds all 4 blocks; D needs 1, 0 free, so D waits
step 26: B reaches 1025 tokens, alone -> AssertionError      (D never ran)

=== D first, then B (4 blocks = 1024 tokens) ===
step 1:  prefill D -> D holds 1 block; B needs 4, 3 free, so B waits
step 21: D finishes -> B is admitted and holds all 4 blocks
step 46: B reaches 1025 tokens, alone -> AssertionError
```

Step 26 is also what the arithmetic predicts: a 1000-token prompt is 1001 tokens after prefill, and 24 decode steps later it is 1025.

### 3. A second path: a prompt that never fits

A prompt longer than the whole cache reaches the same assert without ever going through self-preemption. The prefill loop looks only at the head of `waiting` ([scheduler.py#L31](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L31)). `can_allocate` returns -1 and the loop `break`s ([scheduler.py#L36-L38](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L36-L38)). With nothing running, the decode loop does not run even once, and the assert fires on an empty batch:

```
=== prompt too long (2 blocks = 512 tokens) ===
step 1:  AssertionError        (600-token prompt, never admitted)

=== prompt too long, with others (4 blocks = 1024 tokens) ===
step 1:  prefill A; P (1100 tokens) cannot fit -> break
steps 2-20: only A decodes; C needs 1 block and 3 are free, but it sits behind P
step 21: A finishes, nothing running -> AssertionError      (C never ran)
```

Because the prefill loop never looks past the head of the queue, one request that can never fit also blocks every request behind it (head-of-line blocking).

| | Grows past the cache (#274 as reported) | Prompt never fits |
|---|---|---|
| Admitted? | yes, runs and generates tokens | never |
| Crashes at | the step it outgrows the cache | the first step where nothing else is running |
| Code path | decode loop → `can_append` fails → self-preemption → assert | prefill `break` → decode loop skipped → assert |
| Known up front? | only an upper bound: prompt + `max_tokens` | yes, from the prompt length alone |

## Upstream PR #277

[PR #277](https://github.com/GeeeekExplorer/nano-vllm/pull/277) (open, no reviews as of 2026-09-28) changes one line. It keeps all scheduling behavior and replaces the assert with an explicit error:

```diff
-        assert scheduled_seqs
+        if not scheduled_seqs:
+            raise RuntimeError(
+                f"KV cache exhausted: sequence {seq.seq_id} needs more blocks than "
+                f"num_kvcache_blocks provides in total, even after preempting every "
+                f"other running sequence. Increase num_kvcache_blocks or reduce "
+                f"max_model_len / max_tokens."
+            )
```

I tested it by applying the PR's diff to a clean copy of `bb823b3` and running the same simulation against that copy. For the first scenario, I also caught the error and called `schedule()` again:

```bash
mkdir pr277 && git -C nano-vllm archive bb823b3 | tar -x -C pr277
curl -sL https://github.com/GeeeekExplorer/nano-vllm/pull/277.diff | patch -d pr277 -p1
PYTHONPATH=pr277 python sched_sim.py
```

```
=== B first, then D (4 blocks = 1024 tokens) ===
step 26: RuntimeError: KV cache exhausted: sequence 0 needs more blocks than ...
  running=[] waiting=['B', 'D'] finished=[]
  -> catch it and call schedule() again:
step 27: RuntimeError: (same error)

=== prompt too long (2 blocks = 512 tokens) ===
step 1: RuntimeError: ... even after preempting every other running sequence ...

=== prompt too long, with others (4 blocks = 1024 tokens) ===
step 21: RuntimeError: ...
  running=[] waiting=['P', 'C'] finished=['A']
```

What it gets right:

- It agrees with the analysis above: an empty batch here is permanent, so retrying would hang. The issue suggested retrying; the PR author traced the code and rejected that.
- `raise` still works under `python -O`, where `assert` is stripped.
- The change is small, which fits a minimal teaching codebase.

What it leaves open:

1. **One bad request fails the whole batch.** The error propagates out of `generate()`, which returns only after all requests finish. D, a 10-token request, never runs. A finished, but its output is lost with the exception.
2. **The engine cannot recover.** The request is preempted back to the head of `waiting` before the error is raised, so every later `schedule()` call raises again (step 27). There is no API to drop a request.
3. **It is detected late.** In the first case, 24 generated tokens are thrown away. In the last case, P blocks C for 20 steps before anything is reported, and C fits easily.
4. **Two of the error message's suggestions do not work.** `num_kvcache_blocks` is overwritten at startup from free GPU memory ([model_runner.py#L113](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L113)), so setting it has no effect. `max_model_len` does not limit sequence length. Only reducing `max_tokens`, shortening the prompt, or raising `gpu_memory_utilization` helps. The message also says "even after preempting every other running sequence" when, for a prompt that never fits, nothing was preempted.

A smaller point: the message uses `seq`, a variable left over from whichever loop ran last. For a prompt that never fits, it happens to point at the right request, the head of `waiting` from the prefill loop, but that works by accident.

## Fix

Written and tested in my local checkout, not submitted upstream. The goal: only the request that cannot fit is affected, every other request completes, and no exception is raised. The two paths are handled separately, and a new `finish_reason` tells the caller what happened.

The cache capacity in tokens is `num_kvcache_blocks * kvcache_block_size`. The scheduler computes it once, since `ModelRunner` has already sized the cache by the time the scheduler is created:

```python
# Scheduler.__init__
self.max_seq_tokens = config.num_kvcache_blocks * config.kvcache_block_size
```

**Prompt never fits: reject on arrival.** In `Scheduler.add`, a prompt longer than the whole cache is marked finished instead of queued. It never blocks the requests behind it:

```python
def add(self, seq: Sequence):
    if seq.num_tokens > self.max_seq_tokens:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = "prompt_too_long"
        return
    self.waiting.append(seq)
```

**Grows past the cache: finish it in `postprocess`.** Right after a new token is appended, if the sequence is now longer than the cache, it can never run another step. It finishes with everything it generated, including the token just sampled: that token's logits came from KV that did fit. The check sits next to the existing EOS and `max_tokens` checks and reuses the existing finish path ([scheduler.py#L89-L92](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L89-L92)):

```python
if not seq.ignore_eos and token_id == self.eos:
    seq.finish_reason = "stop"
elif seq.num_completion_tokens == seq.max_tokens:
    seq.finish_reason = "length"
elif len(seq) > self.max_seq_tokens:
    seq.finish_reason = "kv_cache_full"
if seq.finish_reason:
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
```

My first idea was to put this check in `schedule()`, before the self-preemption branch. Working it through showed two problems. Finishing the only running request there still leaves the step with an empty batch. And a request finished inside `schedule()` is not in the batch `step()` returns, so its output would never reach `generate()`. In `postprocess`, both problems go away: the request finishes one step earlier, while it is still in the batch, and a sequence that never outgrows the cache never reaches the self-preemption branch in this state. The decode loop and its `assert` are unchanged. The assert now holds for both paths and stays as a safety net.

**Engine plumbing.**
- `Sequence` gets `finish_reason = None`: `stop`, `length`, `kv_cache_full` or `prompt_too_long`.
- `LLMEngine.add_request` returns the sequence. `generate()` records a request rejected on arrival right away, because it never appears in any step's output. If every request is rejected, the main loop never runs and the results are still returned.
- Each result dict gains a `finish_reason` key. `text` and `token_ids` are unchanged.

vLLM reports "ran past `max_model_len`" as `length`. I use a separate `kv_cache_full` so the cause is visible while learning.

**Compared with PR #277:**

| | PR #277 | My fix |
|---|---|---|
| A request that grows past the cache | `RuntimeError` | finishes with the tokens it generated, `kv_cache_full` |
| A prompt that never fits | `RuntimeError` once nothing else is running; blocks the queue until then | rejected on arrival, `prompt_too_long`; never enters the queue |
| Other requests in the same `generate()` call | lost with the exception | complete normally |
| Engine after the event | every later `schedule()` raises again | keeps working |
| Size of change | 1 file, a few lines | 4 files, a few dozen lines |

**Alternatives I did not take.**
- Rejecting on prompt + `max_tokens` up front would also catch the first path, but it is conservative: it can reject a request that would have stopped at EOS long before running out of room.
- Raising `ValueError` in `generate()` for a prompt that never fits is simpler, and it is what vLLM's offline API does for prompts over `max_model_len`. But then one bad prompt fails the whole batch, which was the first problem with PR #277.

**Limits.**
- Capacity comes from rank 0's KV cache. Under tensor parallelism, that relies on all ranks agreeing on the block count, which is upstream [#187](https://github.com/GeeeekExplorer/nano-vllm/issues/187).
- It does not enforce `max_model_len`, so the CUDA Graph mismatch in [#190](https://github.com/GeeeekExplorer/nano-vllm/issues/190) remains. Capping at `min(capacity, max_model_len)` in the same check would likely address both. I have not tested that.

## Verification

**Scheduler-only simulation** (`sched_sim.py` above). Before is a clean `bb823b3`; after is my local checkout with the fix. The lines are real output, trimmed; the comment in parentheses is mine:

```
=== B first, then D (4 blocks = 1024 tokens) ===
before:  step  1: prefill ['B']
         step 26: AssertionError
after:   step  1: prefill ['B']
         step 25: B finished, kv_cache_full, 25 tokens
         step 26: prefill ['D']
         step 45: D finished, length, 20 tokens
         all finished after 45 steps

=== D first, then B (4 blocks = 1024 tokens) ===
before:  step 20: D finished, -, 20 tokens
         step 21: prefill ['B']
         step 46: AssertionError
after:   step 20: D finished, length, 20 tokens
         step 21: prefill ['B']
         step 45: B finished, kv_cache_full, 25 tokens
         all finished after 45 steps

=== prompt too long (2 blocks = 512 tokens) ===
before:  step  1: AssertionError
after:   step  0: P finished, prompt_too_long, 0 tokens
         all finished after 0 steps

=== prompt too long, with others (4 blocks = 1024 tokens) ===
before:  step  1: prefill ['A']
         step 20: A finished, -, 20 tokens
         step 21: AssertionError       (C never ran)
after:   step  0: P finished, prompt_too_long, 0 tokens
         step  1: prefill ['A', 'C']
         step  5: C finished, length, 5 tokens
         step 20: A finished, length, 20 tokens
         all finished after 20 steps
```

Every step number matches what I predicted before running. For example, B's 1000-token prompt is 1001 tokens after prefill, so it passes 1024 tokens after 24 more decode steps, at step 25.

**Real engine** (Qwen3-0.6B, RTX 4060 Laptop, `enforce_eager=True`):

| Run | Before | After |
|---|---|---|
| `repro_274.py`, 500-token prompt, 2 blocks | `AssertionError` at step 14 | finishes at step 13, `kv_cache_full`, 13 tokens; the trace has no step 14 |
| same, 50000-token prompt | `AssertionError` at step 1 (expected from the simulation's "prompt too long" case; not run on the real engine) | returns at once, `prompt_too_long`, 0 tokens; the trace shows `REJECT` and 0 steps |

**Regression checks:**
- `trace_example.py`, where preemption does work because there is a younger request to evict: still `preemptions: 1`, and all four requests finish with `length`, 64 tokens each. The longest request is about 2350 tokens against a 3840-token capacity, so the new check never fires.
- The two prompts from `example.py`: one ends at EOS (`stop`, 208 tokens), the other at `max_tokens` (`length`, 256 tokens). `text` and `token_ids` are as before, with the new `finish_reason` key added.

## What I learned

- Preemption only works when there is someone else to evict. The self-preemption branch exists, but nothing handles what comes after it.
- Whether an evicted request can come back depends on the cache's total size, not on how much is free right now. Free space fills up again as others finish; the total never grows.
- An empty batch here means the request can never be served, not that the engine should wait. The issue and the PR disagree on this, and tracing the code shows the PR is right.
- The prefill loop only looks at the head of the queue, so one request that can never fit blocks everything behind it.
- Testing a PR against my own scenarios found problems its description does not mention. A scheduler-only simulation made that cheap: seconds per run, no GPU.
- My first guess (that another request could still be running when one outgrows the cache) was wrong. Working out the block arithmetic, then running the simulation, caught it.
- `max_model_len` in nano-vllm only sizes buffers (warmup, CUDA Graph block tables). It is not an enforced limit, which is also behind upstream #190.
- Where a check lives matters as much as what it checks. The same condition in `schedule()` would have left an empty batch and lost the request's output; in `postprocess` it reuses the normal finish path.
- "Don't crash" is not the whole goal. The better question is who pays for the failure: with PR #277 the whole batch does; with the fix, only the request that cannot fit.
- Predicting the step numbers before running the simulation made the results a real test rather than something to read after the fact.
