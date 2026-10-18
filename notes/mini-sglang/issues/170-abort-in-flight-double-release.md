# [mini-sglang #170] Aborting a request whose step is in flight releases it twice

- Upstream issue: https://github.com/sgl-project/mini-sglang/issues/170, which I opened on 2026-10-18 · Fix: [PR #111](https://github.com/sgl-project/mini-sglang/pull/111) (open since 2026-03-25), whose description covers only the EOS case · Related: the crash reported in [a comment on #89](https://github.com/sgl-project/mini-sglang/pull/89#issuecomment-4055333948) (backlog A3), [PR #139](https://github.com/sgl-project/mini-sglang/pull/139) (Ascend port, also defers aborts of in-flight requests)
- Status: reported issue, reproduced for three cases on the real engine; PR #111 fixes all three, so no PR from me. I posted the results on the PR and asked the maintainer to take a look ([comment](https://github.com/sgl-project/mini-sglang/pull/111#issuecomment-6256621180)). No reply yet.
- Commit tested: `9a91cfa`, unmodified upstream; PR #111 at its head `04340ee` (based on `20fcd7f`, 6 commits behind main, merges cleanly)
- Written: 2026-10-18
- Study Q&A (Chinese), including the loop-by-loop walk-through behind this page: [QA-overlap.md](../QA-overlap.md), section 10

<!-- Also add a row to CONTRIBUTIONS.md and to the Contributions table in README.md, and keep its status in sync with this page. -->

## TL;DR

- **The bug:** with overlap scheduling (the default), a step's results are processed one loop iteration after the step is launched. If an abort for a request in that step arrives in between, the abort handler frees the request at once, and processing the step then handles the same request a second time.
- **What the second pass breaks** depends on the step:

  | Case | What happens | Result |
  |---|---|---|
  | Aborted during its prefill; the prompt hit the radix cache | `cache_req` runs twice and unlocks the matched node twice | Crash: `assert node.ref_count >= 0` |
  | Aborted during its prefill; the prompt missed | `cache_req` runs twice; the second run frees the pages the first one just put in the tree | No crash, but those pages are in the tree and on the free list at once, and the node stays locked forever |
  | Aborted during the decode step that samples EOS (the case in #111) | `_free_req_resources` runs twice | Crash: the same assertion |
  | Aborted during any other decode step | The step's result is not a finish, so nothing runs a second time | No problem |

- **How I found it:** doing PR #111 as a read-the-code exercise, I asked whether its double free really needs EOS. In decode it does, but a request whose in-flight step is its prefill gets handled twice too. That path crashes with exactly the traceback reported under load in the #89 comment, which had no issue and no fix.
- **Reproduced deterministically** on the real engine by subclassing the offline `LLM` to inject the abort at the right loop iteration. PR #111 passes all three cases: it frees aborted requests only after the in-flight step has been processed.
- **Not the same bug as #168**, and neither fix covers the other. Both come from the same cost of overlap scheduling: the CPU's bookkeeping runs one step behind the GPU.

## What I learned

- Root cause -> Two code paths release the same request (the abort in receive, and the result processing) and neither knows about the other, so the request is released twice.
- Why does overlap scheduling cause bugs like this? -> The CPU's bookkeeping is one step behind the GPU. Between launching a step and processing it, anything that changes a request's state (like an abort) must know the step is still in flight.
- Same bug, different symptoms -> With a cache hit, the double unlock makes ref_count −1 and crashes; with a miss, the root's unlock does nothing, but the second cache_req frees the request's own pages as duplicates. So test both cases.
- How to fix it -> Defer the abort's release until the in-flight step is processed (#111's pending_aborts), so every lock is matched by exactly one unlock.
- More interesting, how to find this new case? -> Asked whether the bug really needs its condition (EOS). It doesn't in prefill, and that case was the crash someone reported months ago.
- How to reproduce a timing bug without a load test -> Subclass the offline LLM and inject the abort at an exact loop iteration (override _forward and offline_receive_msg). The result is deterministic.
- Although bugs in overlap scheduling, but there are two different bugs, two fixes -> #111 doesn't fix #168 and #154 doesn't fix #170. Check which code each PR actually changes.
- Writing the issue -> We should use the general root cause, not one special case.


## Environment

Windows 11 with WSL2 (Ubuntu 24.04), NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9), driver 617.14. Python 3.12, torch 2.9.1+cu128, flashinfer-python 0.7.0.post1, sgl-kernel 0.3.21. Qwen3-0.6B, attention backend chosen automatically: FlashInfer (`fi`).

## Symptom

In server mode, a streaming request is aborted when the API server notices that its HTTP client has disconnected: it sends `AbortMsg`, and the tokenizer process turns it into `AbortBackendMsg` for the scheduler ([tokenizer/server.py L102-L108](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/tokenizer/server.py#L102-L108)). If that message reaches the scheduler while the request's step is in flight, the scheduler process either dies on `assert node.ref_count >= 0` ([radix_cache.py L119](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/kvcache/radix_cache.py#L119)) or keeps running with a radix cache whose pages are also on the free list. In the second case, the next idle moment would probably crash it too, since `run_when_idle` calls `check_integrity()` ([scheduler.py L78-L80](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L78-L80)); before that, any request that hits that prefix may read KV that another request has overwritten. I have not run this through the HTTP server.

## Reproduce

The offline `LLM` never receives aborts, so the script subclasses it and injects an `AbortBackendMsg` in the loop iteration right after the chosen step of request X is launched. From there on, the scheduler runs the same code as in server mode. Three modes:

- `hit`: X's prefill. Request A first runs the same 101-token prompt, so X matches its first 100 tokens.
- `miss`: X's prefill, with an empty radix cache.
- `eos`: the decode step in which X samples EOS. X is a short chat prompt ("Reply with just the word OK.", thinking off), so it ends after a few tokens. To find that step, the script reads the sampled token right after launch; this waits for the GPU, but the CPU still receives the abort before it processes the step.

```bash
python repro_abort_in_flight.py hit ~/huggingface/Qwen3-0.6B
python repro_abort_in_flight.py miss ~/huggingface/Qwen3-0.6B
python repro_abort_in_flight.py eos ~/huggingface/Qwen3-0.6B
```

| Mode | main `9a91cfa` | PR #111 (`04340ee`) |
|---|---|---|
| `hit` | **`AssertionError`**, through `cache_req(finished=False)` at [scheduler.py L164](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L164) | Radix cache 116 evictable, 0 protected; 46,892 slots free, the same as after A alone. `check_integrity` passes |
| `miss` | No crash, but radix cache 0 evictable, **101 protected**, and **47,008 of 47,008** slots free. `check_integrity` raises `free_pages(47008) + cache_pages(101) != num_pages(47008)` | Radix cache 101 evictable, 0 protected; 46,907 slots free. `check_integrity` passes |
| `eos` | **`AssertionError`**, through `_free_req_resources` at [scheduler.py L161](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L161) | Radix cache 21 evictable, 0 protected; 46,987 slots free. `check_integrity` passes |

The script is in the issue. My lab copy of the hit and miss modes (`abort_prefill.py`) prints each step and gave the same numbers.

## How it happens, loop by loop

One iteration of `overlap_loop` ([scheduler.py L90-L106](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L90-L106)) does three things in this order: ① receive messages, ② schedule and launch the next step, ③ process the previous step's results. A step launched in ② is processed in the next iteration's ③, so the next iteration's ① sits in between.

`cache_req(req, finished)` ([cache.py L55-L79](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L55-L79)) is where a request settles its pages with the radix tree:

1. Insert its first `cached_len` tokens; the tree reports how many were already there.
2. `unlock(req.cache_handle)`, the node locked at admission.
3. Free `[old_handle.cached_len, prefix_len)`: tokens that were not cached at admission but are now, so another request must have inserted them, and this request's copy is a duplicate.
4. If finished, free the tail and leave `cache_handle` as it is; otherwise set `cache_handle` to the new node and lock it.

Running it twice on the same request repeats steps 2 and 3.

**Hit.** A's run left 116 tokens in the tree (the 101-token prompt and 15 generated tokens), unlocked; 46,892 slots free.

| Iteration | Phase | Code | State after |
|---|---|---|---|
| 1 | ② admit | `_try_allocate_one` ([prefill.py L39-L61](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py#L39-L61)) matches 100 tokens, ending at node N; `lock(N)`; one slot for the last prompt token | `N.ref_count` = 1, `X.cache_handle` = N; 46,891 free |
| 1 | ② launch | `complete_one` ([engine.py L200](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py#L200)) runs at launch; `filter_reqs` adds X to the decode set | `X.cached_len` = 101; X in `running_reqs` |
| 2 | ① abort | `decode_manager.abort_req` finds X; `_free_req_resources(X)` → `cache_req(X, finished=True)`: all 101 tokens are cached, X's extra slot is a duplicate and is freed, `unlock(N)` | `N.ref_count` = 0; 46,892 free; `X.cache_handle` still N |
| 2 | ② | nothing left to schedule | |
| 2 | ③ | X's prefill result is not a finish → `cache_req(X, finished=False)` ([L163-L164](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L163-L164)) → `unlock(N)` | `N.ref_count` = −1 → **assertion** |

**Miss.** X matches nothing, so its handle is the root, and `lock`/`unlock` on the root do nothing. X gets 101 slots (46,907 free). In iteration 2, ① inserts the 101 tokens as a new node M, unlocked; the books still balance (101 + 46,907 = 47,008). ③ inserts again, and the tree reports all 101 tokens as cached. Step 3 of `cache_req` reads that as "inserted by another request in the meantime" and frees `page_indices[0:101]` as duplicates: M's own pages, which X put there in ①. Then it locks M. The result is 101 slots both in the tree and free, and a node that nobody will unlock.

**EOS.** When the abort arrives, the CPU does not yet know that the in-flight step samples EOS, so X is still in the decode set and the abort frees it. ③ then sees EOS. The guard against freeing twice checks `finished_reqs` ([L158-L161](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L158-L161)), but that set holds only requests freed by `_process_last_data` itself in the previous iteration, and the abort path never adds to it. So `_free_req_resources(X)` runs again: X's table row is freed a second time, and its handle, the node locked after its prefill, is unlocked a second time.

Hit and miss are the same bug. Which step of the second `cache_req` does the damage depends on the handle: a real node turns the double unlock into a crash, while the root turns it into a no-op and leaves the duplicate free to corrupt the cache.

## Root cause

Two code paths release a request, and they do not know about each other: the abort handler in ① ([scheduler.py L189-L198](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L189-L198)), and the processing of its step in ③. The abort path does not update `cache_handle` and does not record that it freed the request; ③ does not check whether the request was aborted.

Only overlap scheduling has a gap between the two: `normal_loop` processes a step in the iteration that launches it, so an abort always arrives after the step has been handled.

The window is short. The abort has to be received by the first ① after the launch, and that ① comes as soon as the previous ③ is done, which waits only for the step before. So the window is about one GPU step, not the length of the request's own prefill. I first thought otherwise.

## The fix (#111)

PR #111 ([diff](https://github.com/sgl-project/mini-sglang/pull/111/files)), condensed:

```python
# _process_one_msg, abort branch: take the request out of the queues, but do not free it yet
if req_to_free is not None:
    self.pending_aborts.add(req_to_free)        # was: self._free_req_resources(req_to_free)

# overlap_loop and normal_loop, right after self._process_last_data(...)
self._free_pending_aborts(ongoing_data)

def _free_pending_aborts(self, ongoing_data):
    ongoing_uids = {req.uid for req in ongoing_data[0].batch.reqs} if ongoing_data else set()
    for req in self.pending_aborts.copy():
        if req.uid in ongoing_uids:             # still in a step on the GPU: try next iteration
            continue
        if req in self.finished_reqs:           # already freed by _process_last_data (EOS case)
            self.pending_aborts.discard(req)
            continue
        self._free_req_resources(req)           # prefill case: after cache_req(finished=False) ran once
        self.pending_aborts.discard(req)
```

In the prefill case, ③ now runs `cache_req(finished=False)` once, which unlocks N and locks the new node, and the deferred free unlocks the new node: one lock, one unlock each. In the EOS case, ③ frees the request, and the deferred free finds it in `finished_reqs` and skips it. The PR's own test mocks the engine and only counts calls to `_free_req_resources`, so it never reaches the assertion; the real engine shows that the EOS case crashes too.

[PR #139](https://github.com/sgl-project/mini-sglang/pull/139), a large port to Ascend NPUs, adds a similar fence for aborts of in-flight requests. I have not tested it.

## Verification

The table under [Reproduce](#reproduce): on PR #111, all three modes finish, `check_integrity` passes, nothing stays protected, and in the hit mode the cache ends exactly as it was after A alone. On main, the hit traceback matches the #89 report line for line (scheduler.py 131 → 105 → 164, cache.py 72 → 40, radix_cache.py 119).

## Relation to #168 and to the #89 report

- **#111 does not fix [#168](168-radix-dedup-stale-page-table.md), and #142/#154 do not fix this.** PR #111 does not touch `cache.py`, and the line behind #168 came in with `c7f800d` (PR #86), before #111's base. #142 and #154 change step 3 of `cache_req` (remap, then free the duplicates), not the double unlock.
- **The #89 report** goes through `cache_req(finished=False)` at L164, which matches the prefill case, not the EOS case at L161. That run used `sglang.bench_serving`, which as far as I know only disconnects when a request times out or the run is interrupted, so I can't tell whether this is what happened there.

## Open questions

- **Through the HTTP server.** The API server notices a disconnect only when it sends the next chunk of a streaming response, or when Starlette cancels the response ([api_server.py L190-L200](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/server/api_server.py#L190-L200)). It then waits 0.1 s before sending the abort ([L202-L209](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/server/api_server.py#L202-L209)), and the window is about one step. Not tried.
- **A shared node.** If another running request also holds N, the double unlock does not assert; `ref_count` ends one too low, and the assertion fires later, when the last holder unlocks N. Not tested.
- **Other in-flight decode steps.** Nothing is handled twice on the CPU, but the abort releases the table row and the KV pages while the step may still be running; #111 mentions the row being reused this way.
