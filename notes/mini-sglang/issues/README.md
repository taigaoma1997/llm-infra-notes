# mini-sglang issue write-ups

One page per issue I reproduced. Create one with `scripts/new.sh issue mini-sglang <number> <slug> "<title>"`.

| Issue | Title | Status | Write-up |
|---|---|---|---|
| [#168](https://github.com/sgl-project/mini-sglang/issues/168) (backlog A1) | Radix cache dedup frees KV pages that a running request still reads | reported by me 2026-10-17; root cause found, both open PRs (#142, #154) tested and fix it; both PRs now link to it | [168-radix-dedup-stale-page-table.md](168-radix-dedup-stale-page-table.md) |

## Backlog

From a pass over all 55 open issues and PRs, and the closed ones, on 2026-10-07, checked against `9a91cfa` by reading the code. A1 is reproduced (see the table above); the rest are not yet. Most bugs in this repo come as a fix PR without an issue, so the first step is often a minimal reproduction. Suggested order, revised on 2026-10-17 after A1:

1. A variant of A1: two requests with the same prefix prefilled in the same step, with overlap scheduling off. If it also gives a wrong answer, #168 is not specific to overlap scheduling.
2. A2: review PRs #150 and #165 by checking whether each one catches the state #168 leaves behind (a page-table row pointing at free pages).
3. A5 (#111) as a read-the-code exercise: find the bug from the title before reading the diff.
4. Two PRs that are quick to test: [PR #157](https://github.com/sgl-project/mini-sglang/pull/157) (a prompt that is too long is dropped with only a log line, so the client waits forever; the PR answers it with an empty, finished reply) and [PR #160](https://github.com/sgl-project/mini-sglang/pull/160) (CUDA graphs are captured for batch sizes above `--cuda-graph-max-bs`).
5. A4 (#58) together with C1 (#103), then A3 (the `ref_count` assertion).

The other items opened since 2026-10-07 (#156 to #167) are not sorted yet.

**May still be present, reproducible on one GPU**

| ID | Upstream | Problem, as reported | Notes |
|---|---|---|---|
| A1 | [PR #142](https://github.com/sgl-project/mini-sglang/pull/142), [PR #154](https://github.com/sgl-project/mini-sglang/pull/154); now [#168](https://github.com/sgl-project/mini-sglang/issues/168) | When `cache_req` finds that another request already cached part of a prefix, it frees this request's copy of those pages, but an unfinished request's page table still points at them. Later decode steps can read KV that was reused | **Reproduced on 2026-10-17 with a wrong answer, filed as #168; both PRs fix it.** [Write-up](168-radix-dedup-stale-page-table.md). Two nearly identical fixes, neither with a minimal reproduction before #168. The same situation as nano-vllm [#208](https://github.com/GeeeekExplorer/nano-vllm/issues/208)/[#219](https://github.com/GeeeekExplorer/nano-vllm/issues/219), handled differently. Introduced in [`c7f800d`](https://github.com/sgl-project/mini-sglang/commit/c7f800d) (PR #86), which started caching unfinished requests; SGLang writes the cached indices back into the request's row at this point, at least since [v0.3.0](https://github.com/sgl-project/sglang/blob/v0.3.0/python/sglang/srt/mem_cache/radix_cache.py#L121-L147). With overlap scheduling, a request admitted in the step after another one's prefill also misses the prefix. Mechanism: [QA-overlap](../QA-overlap.md) (Chinese) |
| A2 | [#149](https://github.com/sgl-project/mini-sglang/issues/149), [PR #150](https://github.com/sgl-project/mini-sglang/pull/150) | `RadixPrefixCache.check_integrity` is a no-op, so the radix tree is never checked | CPU only. A second implementation, [PR #165](https://github.com/sgl-project/mini-sglang/pull/165), was opened on 2026-10-13 |
| A3 | [PR #89](https://github.com/sgl-project/mini-sglang/pull/89) comment (no issue) | Under load, the scheduler crashes on `assert node.ref_count >= 0`, inside `cache_req(finished=False)` | Reported on 2026-03-13. The maintainer called it a separate issue; no fix in main since |
| A4 | [#58](https://github.com/sgl-project/mini-sglang/issues/58) | Illegal memory access in offline inference with 13 identical long prompts, on an L20 | Not re-tested since the #103 fix. The L20 has the same compute capability (8.9) as my GPU |
| A5 | [PR #111](https://github.com/sgl-project/mini-sglang/pull/111) | With overlap scheduling, a request aborted in the step it finishes is freed twice | Comes with a test |
| A6 | [PR #151](https://github.com/sgl-project/mini-sglang/pull/151) | The API server never removes the state of finished requests | No GPU needed |
| A7 | [#153](https://github.com/sgl-project/mini-sglang/issues/153), [PR #126](https://github.com/sgl-project/mini-sglang/pull/126) | Cannot be used from Open WebUI | In the code, streaming chunks say `"object": "text_completion.chunk"`, `finish_reason` is always `"stop"`, and `usage` is all zeros. PR #126 changes all three |

**Fixed, reproducible on the parent commit**

| ID | Upstream | Problem | Fix |
|---|---|---|---|
| C1 | [#58](https://github.com/sgl-project/mini-sglang/issues/58), [#67](https://github.com/sgl-project/mini-sglang/issues/67), [#89](https://github.com/sgl-project/mini-sglang/pull/89), [#102](https://github.com/sgl-project/mini-sglang/issues/102) | Illegal memory access with overlap scheduling and the FlashInfer backend | [#103](https://github.com/sgl-project/mini-sglang/pull/103) (`20fcd7f`). The maintainer reproduced it on an H200 by adding `torch.cuda._sleep(1_000_000)` before a forward batch |
| C2 | [PR #124](https://github.com/sgl-project/mini-sglang/pull/124) | A radix node created by a split keeps an old timestamp, which affects eviction order | `ba3b55c` |
| C3 | [PR #80](https://github.com/sgl-project/mini-sglang/pull/80) | With `page_size > 1`, eviction counts in the wrong unit | `dae78f6` |

**Performance, measurable on one GPU.** Several of these wait on numbers: [PR #110](https://github.com/sgl-project/mini-sglang/pull/110) (FlashInfer with page size > 1; the maintainer asked for a micro benchmark), [PR #132](https://github.com/sgl-project/mini-sglang/pull/132) (FP8 KV cache; its author measured on an RTX 4060), [PR #143](https://github.com/sgl-project/mini-sglang/pull/143) (`Req.append_host` copies the whole sequence for every token), [PR #147](https://github.com/sgl-project/mini-sglang/pull/147) (the radix kernel is compiled on the first prefix hit), [PR #56](https://github.com/sgl-project/mini-sglang/pull/56) (more CUDA graph batch sizes), [PR #97](https://github.com/sgl-project/mini-sglang/pull/97) (less conservative memory reservation), [PR #155](https://github.com/sgl-project/mini-sglang/pull/155) and [PR #119](https://github.com/sgl-project/mini-sglang/pull/119) (decode first, mixed prefill and decode batches).

**Read only.** Multiple GPUs: [PR #100](https://github.com/sgl-project/mini-sglang/pull/100), [PR #96](https://github.com/sgl-project/mini-sglang/pull/96). Other hardware: [PR #146](https://github.com/sgl-project/mini-sglang/pull/146) with [#129](https://github.com/sgl-project/mini-sglang/issues/129) (RTX 50 series), [#128](https://github.com/sgl-project/mini-sglang/issues/128), [#90](https://github.com/sgl-project/mini-sglang/issues/90), [#120](https://github.com/sgl-project/mini-sglang/issues/120), [#12](https://github.com/sgl-project/mini-sglang/issues/12), PRs [#138](https://github.com/sgl-project/mini-sglang/pull/138) to [#140](https://github.com/sgl-project/mini-sglang/pull/140).

The other 27 open items are feature requests, new models, docs and refactors.
