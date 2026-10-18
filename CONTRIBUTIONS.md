# Contributions

<!-- The same table is in README.md, under Contributions: when a row is added or its status changes, change both. -->

Issues I reproduced, PRs I opened, and upstream discussions I took part in. Newest first.

| Date | Project | Upstream | Type | Status | Write-up |
|---|---|---|---|---|---|
| 2026-10-18 | mini-sglang | [PR #111](https://github.com/sgl-project/mini-sglang/pull/111#issuecomment-6256621180) | PR comment | posted: the #170 results for all three cases, and what the PR's description and mocked test miss | [notes](notes/mini-sglang/issues/170-abort-in-flight-double-release.md) |
| 2026-10-18 | mini-sglang | [#170](https://github.com/sgl-project/mini-sglang/issues/170) | reported issue | found by reading the code for PR #111; reproduced for three cases, one with the same `ref_count` crash and traceback as a report under load in a #89 comment; open PR #111 fixes all three | [notes](notes/mini-sglang/issues/170-abort-in-flight-double-release.md) |
| 2026-10-17 | mini-sglang | [PR #142](https://github.com/sgl-project/mini-sglang/pull/142#issuecomment-6239502457), [PR #154](https://github.com/sgl-project/mini-sglang/pull/154#issuecomment-6239519480) | PR comment | posted: linked each PR to the #168 reproduction | [notes](notes/mini-sglang/issues/168-radix-dedup-stale-page-table.md) |
| 2026-10-17 | mini-sglang | [#168](https://github.com/sgl-project/mini-sglang/issues/168) | reported issue | root cause found, open PRs tested: #142 and #154 both fix it | [notes](notes/mini-sglang/issues/168-radix-dedup-stale-page-table.md) |
| 2026-10-10 | nano-vllm | [#219](https://github.com/GeeeekExplorer/nano-vllm/issues/219) with [#208](https://github.com/GeeeekExplorer/nano-vllm/issues/208) | reproduced issue | root cause found, open PR tested | [notes](notes/nano-vllm/issues/219-same-step-prefix-sharing.md) |
| 2026-10-08 | nano-vllm | [PR #270](https://github.com/GeeeekExplorer/nano-vllm/pull/270#issuecomment-6064191717) | PR comment | posted: the #190 reproduction; the author then added `Fixes #190` | [notes](notes/nano-vllm/issues/190-cuda-graph-block-tables.md#fixes-compared-the-open-prs-tested) |
| 2026-10-06 | triton-windows | [PR #56](https://github.com/triton-lang/triton-windows/pull/56) | docs PR | merged | [notes](notes/triton-windows/issues/56-pytorch-2.6-os-replace.md) |
| 2026-10-03 | nano-vllm | [#190](https://github.com/GeeeekExplorer/nano-vllm/issues/190) | reproduced issue | root cause found, open PRs tested | [notes](notes/nano-vllm/issues/190-cuda-graph-block-tables.md) |
| 2026-10-02 | nano-vllm | [#170](https://github.com/GeeeekExplorer/nano-vllm/issues/170) | reproduced issue | root cause found | [notes](notes/nano-vllm/issues/170-rmsnorm-fp32-residual.md) |
| 2026-09-29 | nano-vllm | [PR #280](https://github.com/GeeeekExplorer/nano-vllm/pull/280) | bug fix PR | PR open | [notes](notes/nano-vllm/issues/274-kv-cache-exhausted-assert.md) |
| 2026-09-29 | nano-vllm | [#279](https://github.com/GeeeekExplorer/nano-vllm/issues/279) | reported issue | PR open (#280) | [notes](notes/nano-vllm/issues/274-kv-cache-exhausted-assert.md) |
| 2026-09-25 | nano-vllm | [#274](https://github.com/GeeeekExplorer/nano-vllm/issues/274) | reproduced issue | PR open (#280) | [notes](notes/nano-vllm/issues/274-kv-cache-exhausted-assert.md) |

**Type:** reproduced issue · reported issue · bug fix PR · docs PR · test PR · feature PR · review · PR comment

**Status:** investigating · root cause found · open PR tested (someone else's fix) · PR open · merged · closed · posted (for comments)
