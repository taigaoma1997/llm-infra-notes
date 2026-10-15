# [nano-vllm #219] Identical prefixes prefilled in the same step get separate blocks, and #208

- Upstream issue: https://github.com/GeeeekExplorer/nano-vllm/issues/219 · The other side of the same change: [#208](https://github.com/GeeeekExplorer/nano-vllm/issues/208) (closed by its reporter) · The change: [#218](https://github.com/GeeeekExplorer/nano-vllm/pull/218) (merged) · Open PR: [#243](https://github.com/GeeeekExplorer/nano-vllm/pull/243)
- Status: root cause found, open PR tested. Reproduced both sides with a scheduler simulation and on the real engine. #208 does not give wrong results in nano-vllm; #219 is a real cost. No PR from me: the open PR #243 already removes the duplicate blocks, and in my tests the outputs stay the same.
- Commit tested: `bb823b3` (current upstream), `f64d821~1` (`44a51af`, the code before #218), and PR #243 (`1704c07`, based on `bb823b3`)
- Written: 2026-10-10 · Updated: 2026-10-11
- Study Q&A (Chinese), with every question I asked along the way: [QA-scheduling.md](../QA-scheduling.md)

<!-- Also add a row to CONTRIBUTIONS.md and to the Contributions table in README.md, and keep its status in sync with this page. -->

## TL;DR

- **The question behind both issues:** when does the prefix cache register a block's hash? The old code registered it as soon as the block was allocated, before the GPU had computed it. Since [#218](https://github.com/GeeeekExplorer/nano-vllm/pull/218), it is registered after the step.
- **#208 (old code):** a later request in the same prefill step could reuse blocks that an earlier request had not computed yet. Its reporter closed it after a commenter argued this cannot give wrong results in nano-vllm.
- **#219 (current code):** two identical prompts prefilled in the same step no longer share. Each computes and stores its own copy, and the second registration overwrites the first, so the first copy can no longer be found.
- **#208 does not give wrong results here.** Within each layer, the whole batch's K/V is written before attention reads it, so the reused blocks are filled by the time they are read. On the real engine, the request that reused same-step blocks generated exactly the same tokens as the request that computed them. This relies on two conditions the code never states ([below](#does-208-give-wrong-results)).
- **Side finding:** two runs of the same prompt can drift apart after a few tokens, and that is not a bug. Changing the attention path or the number of tokens in the step changes which kernels run, which changes bf16 rounding enough to flip a near-tied token ([below](#side-finding-why-two-correct-runs-drift-apart)).
- **The open fix works for memory, not for compute.** [PR #243](https://github.com/GeeeekExplorer/nano-vllm/pull/243) merges B's duplicate blocks into A's after the step. In my tests, B ends up on A's blocks, the copies are freed, the dictionary is no longer overwritten, and the outputs do not change. The tokens B computed twice in that step are not saved ([below](#the-cost-of-219-and-pr-243)).

## What I learned

- How scheduling and prefix cache works in detail, and it can function differently when initialize at differnet time steps. 
- Realized that the kv cache has steps: first computed, then written, this leads to the flexibility that we can add half pre-computed attention calculation.
- The id dictionary will get rewritten by the last time, that's how the A get losts. 
- very small T v.s. argmax: they are different. Even if the temperature small, the bf16's logits accuracy will remove such difference. Also, the random selection when logits are the same also brings different results. 
- Observed that the decoding can be different even using the same prompts -> could be the change of attention path and prefill token counts changes the kernel selection, which impacts the calculation results in bf16's last digits.
- Split the caculation and steps of CPU block and GPU kv cache calculation. 
- Can start from some simulation first, then go to GPU for verification. 

## Background: how nano-vllm shares cached blocks

Many requests start the same way: a shared system prompt, or the earlier turns of a chat. The K/V of an identical prefix is identical, so it only needs to be computed and stored once.

- **Full blocks only.** The KV cache is split into blocks of 256 tokens. Each full block gets a hash of its 256 tokens chained with the previous block's hash, so equal hashes mean the whole prefix up to that block is equal ([`compute_hash`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L35-L41)). A partly filled last block is never shared.
- **One dictionary.** `hash_to_block_id` maps each registered hash to a block. A block used by several sequences counts them in `ref_count`; it goes back to the free list only when that count reaches 0, and even then its K/V and its entry stay until the block is handed to someone else ([`deallocate`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L94-L101), [`_allocate_block`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L43-L51)).
- **Lookup happens only when a sequence is admitted.** The scheduler looks up a new sequence's blocks once, when it first enters a prefill step ([scheduler.py L35-L45](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L35-L45)). Decode never looks up: a sequence that crosses into a new block just takes a free one. The last block of a prompt is always recomputed, so there is at least one token to produce logits from ([L62](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L62)).
- **A hit only skips the uncached tokens.** A sequence with 512 cached tokens sends only its remaining tokens through the model; attention still reads all of its keys, the cached ones through the block table ([`prepare_prefill`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L128-L170)).

"The same step" means the scheduler's prefill loop admitted both sequences one after the other in a single call, so they run in one forward pass ([scheduler.py L30-L52](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L30-L52)).

## Timeline

```
2026-04-21  #208: a later request in the same prefill step can hit blocks that are not computed yet
2026-04-21  a commenter argues this cannot give wrong results in nano-vllm; the reporter closes #208 on 04-23
2026-04-26  #218 ("fix chunked prefill bugs and refactor") is merged as bb823b3:
            hashes are now registered in postprocess, after the step
2026-04-28  #219: identical prefixes in the same step now get separate blocks
2026-06-15  PR #243: merge the duplicates after the step
```

#218 does not say it was meant to fix #208, but it rules #208 out, and #219 is the price.

## Old and new code, one step

The example used throughout: an 8-block cache, and two identical 600-token prompts A and B admitted in the same step. Each needs 3 blocks: two full ones (tokens 0–255 and 256–511, hashes `h0` and `h1`) and a partial one with 88 tokens.

**Old code: register at allocation** ([old `allocate` L79-L81](https://github.com/GeeeekExplorer/nano-vllm/blob/44a51afc8a7f60e75ce3f4ee94632430eff9de65/nanovllm/engine/block_manager.py#L59-L82)). A is allocated blocks 0, 1, 2 and registers `h0 → 0`, `h1 → 1` at once. B, scheduled right after, finds both hashes and reuses blocks 0 and 1. After scheduling, before the GPU has run:

```
           0         1         2         3         4         5
      +---------+---------+---------+---------+---------+---------+
user  |   A B   |   A B   |    A    |    B    |         |         |
K/V   |  empty  |  empty  |  empty  |  empty  |  empty  |  empty  |
ref   |    2    |    2    |    1    |    1    |         |         |
dict->|   h0    |   h1    |         |         |         |         |
      +---------+---------+---------+---------+---------+---------+
```

B only has 88 tokens to compute. But blocks 0 and 1 are still empty: A computes them in the same forward pass. That is #208's concern.

**Current code: register after the step.** Neither A nor B finds anything at scheduling time, so B gets blocks 3, 4, 5 and computes all 600 tokens. In `postprocess`, A registers first and B second; the dictionary assignment is a plain overwrite ([`hash_blocks` L120](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L110-L120)). After `postprocess`:

```
           0         1         2         3         4         5
      +---------+---------+---------+---------+---------+---------+
user  |    A    |    A    |    A    |    B    |    B    |    B    |
K/V   | t0-255  |t256-511 |  t512+  | t0-255  |t256-511 |  t512+  |
ref   |    1    |    1    |    1    |    1    |    1    |    1    |
dict->|         |         |         |   h0    |   h1    |         |
      +---------+---------+---------+---------+---------+---------+
```

Blocks 0 and 3 hold the same K/V, and so do 1 and 4. A is still using 0 and 1, but nothing points to them any more.

Sharing across steps is not affected in either version: blocks registered in one step can be hit by requests admitted in later steps.

## Reproduction 1: the scheduler, without a GPU

A script drives the real `Scheduler` and `BlockManager` with a fake model step (every request gets token 1) and prints the state after `schedule()` and after `postprocess()`:

| | old `f64d821~1` | current `bb823b3` |
|---|---|---|
| A's block table | `[0, 1, 2]` | `[0, 1, 2]` |
| B's block table | `[0, 1, 3]` | `[3, 4, 5]` |
| B's cached / scheduled tokens | 512 / 88 | 0 / 600 |
| free blocks after scheduling | 4 of 8 | 2 of 8 |
| dictionary after scheduling | `{h0: 0, h1: 1}` | empty |
| dictionary after `postprocess` | `{h0: 0, h1: 1}` | `{h0: 3, h1: 4}` |

One more difference shows up after `postprocess`: the old code counts 601 cached tokens, the current code 600. The current code is right. The token just sampled is the 601st, and its K/V is written only in the next decode step, when it goes in as the input ([`prepare_decode` L178-L181](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L178-L181)).

## Does #208 give wrong results?

In the old code, B's first 512 tokens are never computed for B; their K/V comes entirely from blocks 0 and 1. So the question is whether A has written them by the time B reads them.

A and B run in one forward pass: 600 + 88 = 688 tokens. The batch goes up the model one layer at a time ([qwen3.py L180-L181](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/models/qwen3.py#L178-L183)), and every layer has its own copy of each block. Inside layer 10, for example, three kernels run in order on one GPU stream ([attention.py L62-L70](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L62-L70)):

| | Kernel | A | B |
|---|---|---|---|
| 1 | qkv projection | q, k, v for tokens 0–599 | q, k, v for tokens 512–599 |
| 2 | `store_kvcache` | writes layer 10's blocks 0, 1, 2 | writes layer 10's block 3 |
| 3 | attention | reads blocks 0, 1, 2 | reads blocks 0, 1, 3: **0 and 1 were filled in step 2** |

So when B's attention runs in layer 10, layers 0–10 of blocks 0 and 1 are already written; layers 11–27 are still empty, and nothing reads them yet. #208's "the blocks are empty" is only true at scheduling time.

### On the real engine

One engine, two calls: `generate([P, P])`, so A and B share a prefill step, then `generate([P])` for C, which reuses blocks that finished computing long before. P is 600 tokens of English text, each request generates 32 tokens, sampling is greedy, and the script logs every block allocation:

| | old code | current code |
|---|---|---|
| A's blocks | `[0, 1, 2]` | `[0, 1, 2]` |
| B's blocks | `[0, 1, 3]`, 512 tokens reused from A in the same step | `[3, 4, 5]`, nothing reused |
| C's blocks | `[0, 1, 4]` | `[3, 4, 6]`, **B's** blocks |
| A vs B | identical | identical |
| A vs C | identical | differ from token 22 on |

With the old code, the same-step reuse (B), the request that computed the blocks (A), and a normal reuse (C) all generate the same 32 tokens. With the current code, C reuses B's blocks, which is the overwrite from the diagram above happening for real.

### Why it holds, and what it depends on

Neither condition is written down anywhere in the code:

| Condition | What keeps it true in nano-vllm |
|---|---|
| The blocks B reuses are computed in the **same** forward pass that reads them | Only the first sequence of a step can be split across steps, and a split sequence uses up the step's token budget, so nothing is admitted after it ([scheduler.py L42-L43](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L42-L43)). The old code registered a long prompt's later blocks before computing them, but nobody could be admitted in between |
| In every layer, the whole batch's K/V is written **before** attention reads it | `store_kvcache` and attention are separate kernels on the same stream |

The experiment covers one case: two 600-token prompts, no splitting, one GPU. If a scheduler admitted other sequences next to a prompt that is split across steps, the first condition could fail, and a reused block would be read before it is computed, with no error.

## Side finding: why two correct runs drift apart

In the current code, A and C differ from token 22 on, while in the old code all three match. My first guess was the decode batch size (A and B decode together, C alone), but the old run rules that out: there, too, A and B decode as a batch of 2 and C alone, and they match.

The difference is in the prefill step. A's step had 1,200 tokens and nobody reused anything, so attention ran without a block table. C's step had 88 tokens and a cache hit, so attention ran with one. The prefix-cache check is made for the whole batch ([model_runner.py L162](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L162)), which is why A, which reused nothing, still went through the block-table path in the old code.

A 2×2 run on the current code separates the two factors. A's prompt stays the same and A reuses nothing; only the rest of its batch changes. Each cell is compared with A in `[P, P]`:

| A's prefill step | attention without block table | attention with block table |
|---|---|---|
| 688 tokens | differs from token 22 | differs from token 13; all 32 tokens match the old code's A |
| 1,200 tokens | baseline (`[P, P]`) | differs from token 22 |

Both factors matter. A different attention path calls a different kernel. A different number of tokens changes the matrix shapes, and cuBLAS and `torch.compile` pick kernels by shape, much as decode at batch size 1 uses `gemv` and at 16 uses `cutlass` ([#190](190-cuda-graph-block-tables.md#inside-a-recorded-graph)). A different kernel accumulates in a different order, and the last bits of bf16 results change. Wherever two candidate tokens are nearly tied, that is enough to flip the choice, and everything after it differs.

Greedy decoding needs a true argmax here, not just a tiny temperature. nano-vllm forbids temperature 0, and its sampler adds random noise to every row. bf16 logits near 16 are only resolved to 0.125, so exact ties do happen, and with noise two identical rows can then pick different tokens. In one run with temperature 1e-5, A and B happened to pick the same token at a tie; had they not, it would have looked like evidence for #208.

## The cost of #219, and PR #243

1. **Duplicate work and memory.** B recomputes 512 tokens that A computes in the same step, and both copies stay in memory.
2. **Lost reuse.** From the moment B registers, A's blocks cannot be found. Once B finishes and its blocks are reused, the entries are deleted too, and nobody can hit this prefix, even while A is still running.

It matters most for offline batches: `generate()` with 100 prompts that share a long system prompt admits as many as fit in the first step, and none of those share.

[PR #243](https://github.com/GeeeekExplorer/nano-vllm/pull/243) merges after the fact: when B registers a hash that already points to a live block with the same tokens, it switches B to that block and frees B's copy ([15 lines](https://github.com/GeeeekExplorer/nano-vllm/pull/243/files)).

I ran the same two scripts on the PR's head (`1704c07`, based on `bb823b3`). The scheduler simulation, same 8-block example:

| | current `bb823b3` | PR #243 |
|---|---|---|
| B's block table after scheduling | `[3, 4, 5]` | `[3, 4, 5]` |
| B's block table after `postprocess` | `[3, 4, 5]` | `[0, 1, 5]` |
| `ref_count` of blocks 0 and 1 | 1 | 2 |
| free blocks after `postprocess` | 2 of 8 | 4 of 8 |
| dictionary after `postprocess` | `{h0: 3, h1: 4}` | `{h0: 0, h1: 1}` |

On the real engine, A and B generate the same 32 tokens, so switching B to A's blocks after the step leaves B's output unchanged, and the later request C reuses the shared blocks (`[0, 1, 6]`). A and C still drift apart from token 22, exactly as on the current code: that is the bf16 effect above, not the PR.

So PR #243 recovers the memory and the lost reuse. It cannot recover the computation: nothing is shared at scheduling time, so B has already computed all 600 tokens by the time the blocks are merged. I did not review its edge cases.

## Thoughts

1. **When a fact is published matters as much as what it says.** What separates the two versions here is only where the registration happens: at allocation, or in `postprocess`. Publishing early saves work for same-step requests but hands out blocks before their K/V is computed; publishing late is safe but repeats the work.
2. **A correctness argument that rests on scheduler rules should be written next to the code that relies on it.** Nothing in `BlockManager` says that early registration is only safe because a split sequence takes the whole step. This is the same kind of thing as [#190](190-cuda-graph-block-tables.md), seen from the other side: there an assumed limit was never enforced; here the assumed conditions happen to hold.
3. **Keep bookkeeping and data apart.** "B sees A's blocks" happens on the CPU, in the dictionary and block tables, before anything is computed. Whether the data is there when it is read is a separate question about kernel order on the GPU.
4. **To compare two runs, change one thing.** Different kernels and shapes move bf16 rounding enough to flip a near-tie, so outputs that drift apart are not proof of a bug. A and B could be compared token by token because they ran in the same batch, through the same kernels.
5. **Simulate scheduling before touching the GPU.** Driving the real scheduler with a fake model step answered the allocation questions in seconds. The GPU run was only needed for the one question about computed values.
