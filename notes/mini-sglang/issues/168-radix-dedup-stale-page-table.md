# [mini-sglang #168] Radix cache dedup frees KV pages that a running request still reads

- Upstream issue: https://github.com/sgl-project/mini-sglang/issues/168, which I opened on 2026-10-17 · Open fixes: [PR #142](https://github.com/sgl-project/mini-sglang/pull/142) (2026-07-08) and [PR #154](https://github.com/sgl-project/mini-sglang/pull/154) (2026-10-05), nearly identical; neither had an issue or a reproduction, and #154 does not mention #142 · Related: [#149](https://github.com/sgl-project/mini-sglang/issues/149)
- Status: reported issue, root cause found, both open PRs tested. Both fix the reproduction, so no PR from me. I linked both PRs to the issue in comments ([#142](https://github.com/sgl-project/mini-sglang/pull/142#issuecomment-6239502457), [#154](https://github.com/sgl-project/mini-sglang/pull/154#issuecomment-6239519480)). No reply from the maintainer yet.
- Commit tested: `9a91cfa`, unmodified upstream; the PRs at their heads, #142 `afa702c` and #154 `3b1bfb9`
- Written: 2026-10-17
- Study Q&A (Chinese), including how overlap scheduling waits on the GPU and the step-by-step trace behind this page: [QA-overlap.md](../QA-overlap.md)

<!-- Also add a row to CONTRIBUTIONS.md and to the Contributions table in README.md, and keep its status in sync with this page. -->

## TL;DR

- **The bug:** with overlap scheduling (the default), a request can be admitted before another request's prompt, which shares its prefix, is in the radix tree. It misses the prefix and computes it again. When its own prompt is inserted, `cache_req(req, finished=False)` finds the shared part already cached and frees the request's copy of it, but the request's page table still points at those pages ([cache.py L70-L74](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L70-L74)).
- **The effect:** nothing visible until another request is given those pages and writes its KV there. From then on, the first request's attention reads the other request's KV, and its output is silently wrong. No crash: the page numbers are still valid.
- **Reproduced end to end:** A and B share a 300-token table of secret codes, and B asks for Bob's code, which is at token 24. A third request C is held back by the admission check until B's duplicate pages are freed, and then gets them. On main, B answers `' 23043555555'` instead of `' 2093.'`; with overlap scheduling off, with the default KV size, or alone, it answers `' 2093.'`.
- **The fixes:** both open PRs point the request's page-table entries at the tree's pages before freeing the duplicates, and both fix the reproduction. I [filed #168](https://github.com/sgl-project/mini-sglang/issues/168) to tie the two PRs to one reproduction.
- **It took two tries.** The first prompt made the overwrite happen, but the answer did not depend on the overwritten positions, so the output did not change.

## What I learned

- Difference between overlap_loop and normal_loop -> Key: need to wait for GPU finish or not.
- How to understand the overlap loop mechanism? -> Instead of following the loop within the code, we can rethink of the loop starting from sending the GPU request. 
- How many multi GPU? -> the scheduler in each gpu only waits for itself, and beween multiple GPUs, there is a sync on all_reduce.
- Root cause of this issue -> When the duplcate table was released, its block table still points to the previous location, which leads to error if this cache was used later.
- This issue needs to show up only when we run out of kv cache, since the release cache was appended to the end of the list; It also do not reproduce using the same batch since these tokens are preserved.
- How these existing PR solved the issue -> create a clone and release it from here, and refreshed the table. 
- We can add an issue to these solved PRs for community contributions.

## Environment

Windows 11 with WSL2 (Ubuntu 24.04), NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, compute capability 8.9), driver 617.14. Python 3.12, torch 2.9.1+cu128, flashinfer-python 0.7.0.post1, sgl-kernel 0.3.21. Qwen3-0.6B, attention backend chosen automatically: FlashInfer (`fi`).

## Symptom

A request that shares a prefix with an earlier request answers wrong. In my reproduction this needs overlap scheduling on (see [Open questions](#open-questions) for a case that might not), and it only shows when the freed pages are reused before the request finishes. With mini-sglang's default KV size on my GPU (47,008 slots), the freed pages sit at the end of a long free list and are not reused in a short run, so the same bug leaves the output unchanged.

## Reproduce

The script sets 740 KV slots and a 350-token prefill budget. A and B are 350 tokens each, the same for the first 300; C is an unrelated 200-token prompt.

```bash
python repro_a1.py ~/huggingface/Qwen3-0.6B
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python repro_a1.py ~/huggingface/Qwen3-0.6B
```

| Code | A, expected `4817` | B, expected `2093` |
|---|---|---|
| main `9a91cfa` | `' 4817.\nAnswer: 481'` | **`' 23043555555'`** |
| main, overlap scheduling off | `' 4817.\nAnswer: 481'` | `' 2093.\nAnswer: 209'` |
| main, default KV size | `' 4817.\nAnswer: 481'` | `' 2093.\nAnswer: 209'` |
| main, B alone | | `' 2093.\nAnswer: 209'` |
| PR #142 (`afa702c`) | `' 4817.\nAnswer: 481'` | `' 2093.\nAnswer: 209'` |
| PR #154 (`3b1bfb9`) | `' 4817.\nAnswer: 481'` | `' 2093.\nAnswer: 209'` |

The default-KV and B-alone rows come from the same prompts run through my lab copy of the script (`recall.py`), which has those two extra modes.

<details>
<summary>repro_a1.py</summary>

```python
"""A running request keeps reading KV pages that radix-cache dedup has freed.

Requests A and B share a 300-token prefix: a table of secret codes. B asks for
Bob's code, which sits in the first 30 tokens. C is unrelated and arrives later.
With 740 KV slots and a 350-token prefill budget, overlap scheduling makes B
compute the shared prefix again, free its own copy when it inserts the prefix,
and keep reading those pages after C has taken them.

Usage:
    python repro_a1.py [model]                                        # overlap scheduling (default)
    MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python repro_a1.py [model]   # for comparison
"""
import sys

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
         "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose", "Sam", "Tina",
         "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zoe", "Adam", "Bella", "Chris", "Diana"]
CODES = [4817, 2093, 7364, 5128, 9450, 3672, 8215, 1946, 6083, 2759,
         5391, 8642, 1207, 7735, 4486, 9921, 3158, 6670, 2304, 8849,
         5512, 1093, 7726, 3381, 9067, 4625, 2918, 6354, 8170, 5039]


def build_prompts(tokenizer):
    def ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def question(first, name):  # 50 tokens; `first` makes A and B differ from token 300 on
        q = ids(f"\nQuestion: What is the secret code of {name}?\nAnswer: The secret code of {name} is")
        pad = ids(" Please read the table carefully before you answer the question." * 5)
        return ids(first) + pad[: 50 - len(ids(first)) - len(q)] + q

    table = "Below is a table of secret codes.\n" + "".join(
        f"The secret code of {n} is {c}.\n" for n, c in zip(NAMES, CODES))
    shared = ids(table)[:300]
    a = shared + question("\n\n", "Alice")
    b = shared + question("\n", "Bob")
    c = ids("The weather in the mountains changes quickly, so hikers should always carry a jacket. " * 20)[:200]
    assert len(a) == len(b) == 350 and len(c) == 200 and a[300] != b[300] and c[0] != a[0]
    return a, b, c


def main():
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    llm = LLM(model, num_page_override=740, max_extend_tokens=350)
    a, b, c = build_prompts(llm.tokenizer)
    long, short = SamplingParams(max_tokens=12, ignore_eos=True), SamplingParams(max_tokens=5, ignore_eos=True)
    outs = llm.generate([a, b, c], [long, long, short])
    print("A, expected 4817:", repr(outs[0]["text"]))
    print("B, expected 2093:", repr(outs[1]["text"]))


if __name__ == "__main__":
    main()
```

</details>

## How it happens, step by step

KV pages are slot numbers (page size 1). The table follows the GPU: while the GPU computes step k, the CPU first processes step k−1's result, then schedules step k+1 and launches it, then waits for step k ([`overlap_loop`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L83-L106)).

| GPU computes | CPU processes the previous step | CPU schedules the next step |
|---|---|---|
| (nothing yet) | | Step 1: A is admitted, matches 0, gets pages 0–349 |
| Step 1: A's prefill | | Step 2: B is admitted. **A's prompt is not in the tree yet**, so B matches 0 and gets pages 350–699 |
| Step 2: B's prefill. The shared 300 tokens now have two copies: pages 0–299 and 350–649 | Step 1: A's prompt is inserted into the tree | Step 3: C arrives but fails the admission check (it needs 227 slots, 40 are free). Decode A and B |
| Step 3: decode | Step 2: B's prompt is inserted. The tree already has its first 300 tokens, so **B's pages 350–649 are freed. B's page table still maps positions 0–299 to them** | Step 4: C now fits (225 needed, 338 available) and gets pages 702–739 and **350–511** |
| Step 4: C's prefill **writes C's KV into pages 350–511** | Step 3 | Step 5: decode A, B and C |
| Step 5 onwards: B's attention reads C's KV at positions 0–161, where Bob's line is | | |

B's first two output tokens come from steps 2 and 3, before C writes, so they are right; the third is wrong. A is never affected: it reads the tree's pages 0–349, which nobody writes.

Without overlap scheduling, step 1's result is processed before step 2 is scheduled, so B matches A's 300 tokens in the tree, and nothing is ever freed.

**Why step 3 itself is safe.** The GPU never reads the radix tree or the free list. A step reads the KV data in the pages, and two lists of page numbers that are copied out of the page table when the step is prepared: where to write the new KV ([scheduler.py L210](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L210)) and which pages attention reads ([fi.py L215](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/attention/fi.py#L215)). Freeing a page only puts its number back on the free list; the data stays until someone else writes it, and that can only happen in a later step, which the GPU runs after this one. The stale entries are copied into every step of B prepared after the free, starting with step 5 (step 4 is C's prefill, without B).

## Why a third request is needed

My first idea was to shrink the KV cache so that A's and B's own decode steps would be given the freed pages. That cannot work, because of the admission check ([prefill.py L50](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/prefill.py#L50)): a request is admitted only if its uncached tokens plus `max_tokens`, plus what the running requests may still generate, fit in the free and evictable slots.

- In this example, for B to be admitted right after A (the case that triggers the bug), the cache needs at least 699 + 2 × `max_tokens` slots. B is counted at full length, so its duplicate copy is already paid for, and A's and B's decode steps always find enough free slots before the freed ones, which go to the end of the free list.
- With fewer slots, B waits until A's prompt is in the tree, matches it, and nothing is freed.

So a third request has to take the freed pages. The admission check does the timing for us: with 740 slots, C (200 tokens, `max_tokens` 5) does not fit while B's duplicate copy is still allocated, and fits as soon as it is freed. The free list then holds 38 never-used slots followed by the 300 freed ones, so C's 200 pages are 702–739 and 350–511.

## The first attempt changed nothing

My first prompts were `[7] × 300` plus `[1] × 50` and `[7] × 300` plus `[2] × 50`. With the same scheduling, C overwrote B's pages exactly as above, but every run, including B alone, gave the same output: `2` repeated. The model was continuing the 50 copies of `2` at the end of B's prompt, which were never overwritten, so it did not matter what the first 300 positions held. To make the overwrite visible, the answer has to depend on the overwritten positions, hence the table of secret codes with Bob's line near the start.

## Root cause

```python
cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
self.unlock(old_handle)
# this part is already in the prefix cache, free it
self._free(page_indices[old_handle.cached_len : cached_len])   # frees B's pages 350–649
...
req.cache_handle = new_handle      # B now holds a lock on the tree's pages 0–299
self.lock(new_handle)              # but its page table still points at 350–649
```

[cache.py L67-L79](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L67-L79). `old_handle.cached_len` is what B matched when it was admitted (0); `cached_len` is what `insert_prefix` found already in the tree (300). The range between them is the duplicate.

This line was written for finished requests. Before [`c7f800d`](https://github.com/sgl-project/mini-sglang/commit/c7f800d5e8654703683136e87d4fd42cfba13852) ([PR #86](https://github.com/sgl-project/mini-sglang/pull/86), 2026-02-26), only `free_and_cache_finished_req` inserted into the tree, and a finished request never reads its page table again. That commit added `cache_req(finished=False)`, which inserts a request's prompt as soon as its prefill is done ([scheduler.py L163-L164](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py#L163-L164)), and kept the same freeing line. SGLang, after inserting, looks the prefix up again and writes the tree's page numbers back into the request's row ([radix_cache.py L531-L534](https://github.com/sgl-project/sglang/blob/e9400c5a8f3a0d0312a37aecef694f5edfd69341/python/sglang/srt/mem_cache/radix_cache.py#L531-L534), in `checkpoint`); mini-sglang has no such step.

## The fixes compared

#154 ([diff](https://github.com/sgl-project/mini-sglang/pull/154/files)):

```python
dup_slice = slice(old_handle.cached_len, cached_len)               # positions 0–299
if dup_slice.start < dup_slice.stop:
    dup_indices = page_indices[dup_slice].clone()                  # 350–649, copied
    if not finished:
        page_indices[dup_slice].copy_(new_handle.get_matched_indices()[dup_slice])   # row now points at 0–299
    self._free(dup_indices)                                        # frees 350–649
```

#142 ([diff](https://github.com/sgl-project/mini-sglang/pull/142/files)) does the same, but clones only when it remaps (unfinished request, non-empty duplicate range). The two behave the same: a finished request's row is not changed before the pages are actually freed.

**Why the clone is needed.** `cache_req` runs inside `lazy_free_region` ([cache.py L93-L104](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/cache.py#L93-L104)), where `_free` only appends `indices[::page_size]`, a view of the page table, to a list, and reads the numbers after the whole batch is processed. Without the clone, the remap would change what that view sees, and the batch would free the tree's pages 0–299, which A and B are both using, and leak 350–649. On three slots:

```
row = [350, 351, 352, 650], tree pages = [0, 1, 2], free list = [700, 701]
with clone:    free list after the batch = [700, 701, 350, 351, 352]
without clone: free list after the batch = [700, 701, 0, 1, 2]
```

After either fix, B's row maps positions 0–299 to the tree's pages, which are locked by `self.lock(new_handle)` while B runs, so they cannot be evicted, and nobody writes them.

## Verification

The table under [Reproduce](#reproduce): on both PRs, B answers `' 2093.\nAnswer: 209'`, the same text as with overlap scheduling off and as B alone, and A and C are unchanged. The bug reproduces identically, token for token, on unmodified upstream and on my traced copy, so the tracing does not cause it.

## Open questions

- **Same-step prefill, without overlap scheduling.** If two requests with the same prefix are prefilled in the same step, both miss, and the second insert should free the second request's copy in the same way. Not tested yet; if it holds, the bug is not specific to overlap scheduling.
- **[#149](https://github.com/sgl-project/mini-sglang/issues/149) and its fix [PR #165](https://github.com/sgl-project/mini-sglang/pull/165)** add a real `check_integrity` for the radix tree. #149 lists this bug as related. Whether the check would catch it, since the stale entries are in the page table rather than in the tree, is something to find out when reviewing #165.
