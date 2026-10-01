# nano-vllm issue write-ups

One page per issue I reproduced. Create one with `scripts/new.sh issue nano-vllm <number> <slug> "<title>"`.

| Issue | Title | Status | Write-up |
|---|---|---|---|
| [#274](https://github.com/GeeeekExplorer/nano-vllm/issues/274) | Engine crashes on `assert scheduled_seqs` when one sequence outgrows the KV cache | PR open ([#280](https://github.com/GeeeekExplorer/nano-vllm/pull/280)) | [274](274-kv-cache-exhausted-assert.md) |
| [#279](https://github.com/GeeeekExplorer/nano-vllm/issues/279) (reported by me) | Prompt longer than the whole KV cache blocks the queue and crashes | PR open ([#280](https://github.com/GeeeekExplorer/nano-vllm/pull/280)) | [274, Case 2](274-kv-cache-exhausted-assert.md#the-failing-cases) |

## Backlog

From a pass over all 111 issues on 2026-09-25, checked against commit `bb823b3`. Suggested order: #274, #170, #190, #219, #240, the fixed bugs, then #175.

**Still present, reproducible on one GPU**

| Issue | Problem | Upstream PR |
|---|---|---|
| [#170](https://github.com/GeeeekExplorer/nano-vllm/issues/170) | `RMSNorm` modifies an fp32 input in place, which corrupts the residual | [#169](https://github.com/GeeeekExplorer/nano-vllm/pull/169), [#205](https://github.com/GeeeekExplorer/nano-vllm/pull/205) |
| [#190](https://github.com/GeeeekExplorer/nano-vllm/issues/190) / [#106](https://github.com/GeeeekExplorer/nano-vllm/issues/106) | CUDA Graph replay fails with a `block_tables` shape mismatch once a sequence passes `max_model_len` | [#191](https://github.com/GeeeekExplorer/nano-vllm/pull/191) |
| [#261](https://github.com/GeeeekExplorer/nano-vllm/issues/261) | Cannot start on Windows (no NCCL, no libuv); already worked around locally | [#267](https://github.com/GeeeekExplorer/nano-vllm/pull/267) |

**Design questions, visible in a trace**

| Issue | Question | Upstream PR |
|---|---|---|
| [#219](https://github.com/GeeeekExplorer/nano-vllm/issues/219) | Identical prefixes prefilled in the same step get separate blocks. Measured on `bb823b3` (scheduler simulation, 8-block cache): two identical 600-token prompts in one step leave 2 blocks free, so each took 3 blocks instead of sharing 2. This is the other side of the #208 fix | [#243](https://github.com/GeeeekExplorer/nano-vllm/pull/243) |
| [#240](https://github.com/GeeeekExplorer/nano-vllm/issues/240) | Does `may_append` allocate one token too late? (Probably not: see [#150](https://github.com/GeeeekExplorer/nano-vllm/issues/150).) | |
| [#175](https://github.com/GeeeekExplorer/nano-vllm/issues/175) | CUDA Graph decode still allocates tensors every step (performance) | [#176](https://github.com/GeeeekExplorer/nano-vllm/pull/176), [#253](https://github.com/GeeeekExplorer/nano-vllm/pull/253) |

**Fixed in `f64d821`, reproducible on `f64d821~1`**

| Issue | Problem |
|---|---|
| [#114](https://github.com/GeeeekExplorer/nano-vllm/issues/114) / [#144](https://github.com/GeeeekExplorer/nano-vllm/issues/144) | A prompt that is a whole number of blocks and fully cached gives `seqlen_q = 0` and crashes. Measured (scheduler simulation, two identical 512-token prompts in different steps): on `f64d821~1` the second one hits both blocks and has nothing left to compute; on `bb823b3` it reuses one block and recomputes the last. The crash itself happens inside the model, so it needs the real engine |
| [#163](https://github.com/GeeeekExplorer/nano-vllm/issues/163) | A recycled block keeps its old hash, so a later request can hit stale content |
| [#208](https://github.com/GeeeekExplorer/nano-vllm/issues/208) | A request can hit blocks that an earlier request in the same prefill batch has not written yet. Measured on `f64d821~1` with the #219 scenario: 4 blocks free, so the second prompt shared blocks that had not been computed yet. The fix (register hashes in `postprocess`) is what causes #219 |

**Questions that test understanding:** [#20](https://github.com/GeeeekExplorer/nano-vllm/issues/20), [#143](https://github.com/GeeeekExplorer/nano-vllm/issues/143) (chained hashes), [#30](https://github.com/GeeeekExplorer/nano-vllm/issues/30), [#91](https://github.com/GeeeekExplorer/nano-vllm/issues/91) (`can_append`), [#115](https://github.com/GeeeekExplorer/nano-vllm/issues/115) (partly filled blocks), [#107](https://github.com/GeeeekExplorer/nano-vllm/issues/107) (CUDA Graph buffer sizes), [#80](https://github.com/GeeeekExplorer/nano-vllm/issues/80) (warmup size), [#155](https://github.com/GeeeekExplorer/nano-vllm/issues/155) with [#246](https://github.com/GeeeekExplorer/nano-vllm/issues/246) (shared-memory RPC).

**Needs more than one GPU (read only):** [#99](https://github.com/GeeeekExplorer/nano-vllm/issues/99), [#125](https://github.com/GeeeekExplorer/nano-vllm/issues/125), [#144](https://github.com/GeeeekExplorer/nano-vllm/issues/144) part 1, [#187](https://github.com/GeeeekExplorer/nano-vllm/issues/187), [#246](https://github.com/GeeeekExplorer/nano-vllm/issues/246).
