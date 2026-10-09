# mini-sglang issue write-ups

One page per issue I reproduced. Create one with `scripts/new.sh issue mini-sglang <number> <slug> "<title>"`.

| Issue | Title | Status | Write-up |
|---|---|---|---|
| | | | |

## Backlog

From a pass over all 55 open issues and PRs, and the closed ones, on 2026-10-07, checked against `9a91cfa` by reading the code. Nothing here is reproduced yet. Most bugs in this repo come as a fix PR without an issue, so the first step is often a minimal reproduction. Suggested order: #142/#154, #149, #58 together with #103, the `ref_count` assertion, then #111.

**May still be present, reproducible on one GPU**

| Upstream | Problem, as reported | Notes |
|---|---|---|
| [PR #142](https://github.com/sgl-project/mini-sglang/pull/142), [PR #154](https://github.com/sgl-project/mini-sglang/pull/154) (no issue) | When `cache_req` finds that another request already cached part of a prefix, it frees this request's copy of those pages, but an unfinished request's page table still points at them. Later decode steps can read KV that was reused | Two nearly identical fixes, neither with a minimal reproduction. The same situation as nano-vllm [#208](https://github.com/GeeeekExplorer/nano-vllm/issues/208)/[#219](https://github.com/GeeeekExplorer/nano-vllm/issues/219), handled differently. Introduced in [`c7f800d`](https://github.com/sgl-project/mini-sglang/commit/c7f800d) (PR #86), which started caching unfinished requests; SGLang writes the cached indices back into the request's row at this point, at least since [v0.3.0](https://github.com/sgl-project/sglang/blob/v0.3.0/python/sglang/srt/mem_cache/radix_cache.py#L121-L147). With overlap scheduling, a request admitted in the step after another one's prefill also misses the prefix. Mechanism: [QA-overlap](../QA-overlap.md) (Chinese) |
| [#149](https://github.com/sgl-project/mini-sglang/issues/149), [PR #150](https://github.com/sgl-project/mini-sglang/pull/150) | `RadixPrefixCache.check_integrity` is a no-op, so the radix tree is never checked | CPU only |
| [PR #89](https://github.com/sgl-project/mini-sglang/pull/89) comment (no issue) | Under load, the scheduler crashes on `assert node.ref_count >= 0`, inside `cache_req(finished=False)` | Reported on 2026-03-13. The maintainer called it a separate issue; no fix in main since |
| [#58](https://github.com/sgl-project/mini-sglang/issues/58) | Illegal memory access in offline inference with 13 identical long prompts, on an L20 | Not re-tested since the #103 fix. The L20 has the same compute capability (8.9) as my GPU |
| [PR #111](https://github.com/sgl-project/mini-sglang/pull/111) | With overlap scheduling, a request aborted in the step it finishes is freed twice | Comes with a test |
| [PR #151](https://github.com/sgl-project/mini-sglang/pull/151) | The API server never removes the state of finished requests | No GPU needed |
| [#153](https://github.com/sgl-project/mini-sglang/issues/153), [PR #126](https://github.com/sgl-project/mini-sglang/pull/126) | Cannot be used from Open WebUI | In the code, streaming chunks say `"object": "text_completion.chunk"`, `finish_reason` is always `"stop"`, and `usage` is all zeros. PR #126 changes all three |

**Fixed, reproducible on the parent commit**

| Upstream | Problem | Fix |
|---|---|---|
| [#58](https://github.com/sgl-project/mini-sglang/issues/58), [#67](https://github.com/sgl-project/mini-sglang/issues/67), [#89](https://github.com/sgl-project/mini-sglang/pull/89), [#102](https://github.com/sgl-project/mini-sglang/issues/102) | Illegal memory access with overlap scheduling and the FlashInfer backend | [#103](https://github.com/sgl-project/mini-sglang/pull/103) (`20fcd7f`). The maintainer reproduced it on an H200 by adding `torch.cuda._sleep(1_000_000)` before a forward batch |
| [PR #124](https://github.com/sgl-project/mini-sglang/pull/124) | A radix node created by a split keeps an old timestamp, which affects eviction order | `ba3b55c` |
| [PR #80](https://github.com/sgl-project/mini-sglang/pull/80) | With `page_size > 1`, eviction counts in the wrong unit | `dae78f6` |

**Performance, measurable on one GPU.** Several of these wait on numbers: [PR #110](https://github.com/sgl-project/mini-sglang/pull/110) (FlashInfer with page size > 1; the maintainer asked for a micro benchmark), [PR #132](https://github.com/sgl-project/mini-sglang/pull/132) (FP8 KV cache; its author measured on an RTX 4060), [PR #143](https://github.com/sgl-project/mini-sglang/pull/143) (`Req.append_host` copies the whole sequence for every token), [PR #147](https://github.com/sgl-project/mini-sglang/pull/147) (the radix kernel is compiled on the first prefix hit), [PR #56](https://github.com/sgl-project/mini-sglang/pull/56) (more CUDA graph batch sizes), [PR #97](https://github.com/sgl-project/mini-sglang/pull/97) (less conservative memory reservation), [PR #155](https://github.com/sgl-project/mini-sglang/pull/155) and [PR #119](https://github.com/sgl-project/mini-sglang/pull/119) (decode first, mixed prefill and decode batches).

**Read only.** Multiple GPUs: [PR #100](https://github.com/sgl-project/mini-sglang/pull/100), [PR #96](https://github.com/sgl-project/mini-sglang/pull/96). Other hardware: [PR #146](https://github.com/sgl-project/mini-sglang/pull/146) with [#129](https://github.com/sgl-project/mini-sglang/issues/129) (RTX 50 series), [#128](https://github.com/sgl-project/mini-sglang/issues/128), [#90](https://github.com/sgl-project/mini-sglang/issues/90), [#120](https://github.com/sgl-project/mini-sglang/issues/120), [#12](https://github.com/sgl-project/mini-sglang/issues/12), PRs [#138](https://github.com/sgl-project/mini-sglang/pull/138) to [#140](https://github.com/sgl-project/mini-sglang/pull/140).

The other 27 open items are feature requests, new models, docs and refactors.
