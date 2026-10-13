# 001 · When does nano-vllm preempt, with 1000 requests and a full KV cache?

> Written: 2026-10-12 · Status: done

## Question

1000 requests arrive at once, each with a 500-token prompt and 300 tokens to generate. The KV cache holds 1024 blocks of 256 tokens. Step by step, what does nano-vllm's scheduler do: how many prefill steps, when does it preempt, and how much work does preemption add?

## Setup

| | |
|---|---|
| GPU | none used: only the scheduler runs, on the CPU |
| Software | Python 3.10.21; nano-vllm [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8) (full list in [results/env.txt](results/env.txt)) |
| Model | none: each scheduled sequence gets token 1 per step |
| Workload | 1000 requests, 500-token prompts with distinct tokens (no prefix sharing), `max_tokens=300`, `ignore_eos=True` |
| Scheduler settings | nano-vllm's defaults: `max_num_seqs=512`, `max_num_batched_tokens=16384`, block size 256. KV cache: 1024 blocks, then 2048 and 4096 |

## Method

[preempt_sim.py](preempt_sim.py) builds nano-vllm's real `Scheduler` (with its `BlockManager`) and calls `schedule()` and `postprocess()` in a loop, the way `LLMEngine.step()` does, but with no model in between. Each line of the output merges consecutive steps that look alike. Scheduling depends only on token counts, so a fake token gives the same steps as the real model would here (with `ignore_eos`, nothing ends early).

1024 blocks is a number picked to make the cache run out. For Qwen3-0.6B one block takes 28 MiB, so 1024 blocks would need 28 GiB ([KV cache sizing](../../notes/fundamentals/01-kv-cache-sizing.md)).

## Results

### 1024 blocks

The first 330 steps ([full output](results/output.txt)). `free` is counted right after `schedule()`; `running` and `waiting` after `postprocess()`.

| Steps | Kind | Sequences per step | What happens | Free / running / waiting |
|---|---|---|---|---|
| 1–16 | prefill | 32 | 32 × 500 = 16,000 tokens per step. The 33rd prompt does not fit in the 384 tokens left, and only the first sequence of a step may be split | 0 / 512 / 488 |
| 17–28 | decode | 512 | Lengths 501 to 512: all still in their second block. Each step tries prefill first and fails: the head of the queue needs 2 blocks, none is free | 0 / 512 / 488 |
| 29 | decode | 341 | Length 513: every sequence needs a third block. 171 sequences are preempted, the most recently admitted first; each frees 2 blocks, enough for two others (341 + 171 = 512) | 1 / 341 / 659 |
| 30–284 | decode | 341 | The head of the waiting queue is now a preempted sequence of 513 tokens; it needs 3 blocks and only 1 is free | 1 / 341 / 659 |
| 285 | decode | 256 | Length 769: a fourth block each. 85 more preempted, 3 blocks each | 0 / 256 / 744 |
| 286–315 | decode | 256 | At step 315 all 256 reach 300 generated tokens and finish together | 1024 / 0 / 744 after the step |
| 316–325 | prefill | 21–32 | The preempted sequences come back first, because preemption puts them at the front of the queue. They recompute their prompt and the tokens they had generated: first the 85 of 769 tokens (21 × 769 = 16,149 tokens per step), then the 171 of 513 | |
| 326–328 | prefill | 32, then 6 | New requests, until the blocks run out again | 1 / 341 / 403 |
| 329– | decode | 341 | The same pattern repeats | |

Summary: 1030 steps (56 prefill, 974 decode), 569 preemptions, 840,950 prefill tokens for 500,000 tokens of prompts.

### A larger cache

| KV cache | Steps | Preemptions | Prefill tokens | Most sequences in `running` |
|---|---|---|---|---|
| 1024 blocks | 1030 | 569 | 840,950 | 512 |
| 2048 blocks | 645 | 488 | 744,488 | 1000 |
| 4096 blocks | 630 | 0 | 500,000 | 1000 |

([2048 output](results/output_2048.txt), [4096 output](results/output_4096.txt))

With 2048 blocks there are still 488 preemptions. All 1000 sequences get in (32 prefill steps, 2 blocks each), but a decode step serves only the first 512 of `running`, so the other 488 hold their blocks without generating anything. At step 45 the first 512 cross into their third block, and the sequences preempted to make room are exactly those 488 idle ones, at the tail of `running`. They come back after the first group finishes and recompute their prompts: 244,488 extra prefill tokens.

With 4096 blocks nothing is preempted, and the run is the shortest. It still takes two rounds of 299 decode steps, because only 512 sequences decode at a time.

## Conclusion

Prefill keeps going step after step until the KV cache is full; only then does decoding start. When every running sequence crosses a block boundary in the same step, the scheduler preempts about a third of them (or a quarter, at the next boundary), and they wait until a whole group finishes. In this workload preemption adds 68% more prefill work with 1024 blocks.

A second finding: `max_num_seqs` limits the sequences per step, not the sequences in `running`. When the cache can hold more than 512 sequences, the extra ones take blocks but do not decode until the first 512 finish, and they are the first to be preempted. Open PR [#271](https://github.com/GeeeekExplorer/nano-vllm/pull/271) proposes counting the sequences already in `running` when admitting new ones.

## Caveats

- Every request has the same length, so all of them cross block boundaries in the same step. Real prompts and outputs vary, which spreads the preemptions out.
- Steps are not time. A prefill step of 16,000 tokens and a decode step of 512 tokens take very different times on a GPU, so 1030 against 630 steps says nothing about how much slower the first run is.
- The simulation stops at the scheduler; the model and the cache contents are not involved.

## Reproduce

```bash
NANO_VLLM=/path/to/nano-vllm PYTHON=/path/to/python bash run.sh
```

`NANO_VLLM` is a nano-vllm checkout at `bb823b3`; `PYTHON` is an interpreter in which nano-vllm imports (it imports torch and flash-attn, though nothing runs on the GPU). `run.sh` writes the three outputs and `env.txt` to `results/`. One run with other numbers: `python preempt_sim.py <requests> <prompt length> <max_tokens> <blocks>`.
