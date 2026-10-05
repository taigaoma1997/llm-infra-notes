# Roadmap

Budget: about 10 hours a week alongside a full-time job. Phases overlap. Dates are targets, and I review this page once a month.

## Goal for March 2027

- A small inference engine I wrote myself, benchmarked against nano-vllm
- At least two merged PRs to SGLang or Miles
- Notes that explain the core ideas of a modern inference engine in my own words

## Phase 1 · Engine internals with nano-vllm (Sep to Oct 2026) · in progress

- [x] Run nano-vllm end to end
- [ ] Notes 01 to 04: request lifecycle, scheduler, block manager and prefix cache, model runner and CUDA Graph
- [ ] Reproduce two or three open issues and write up each one
- [ ] Fundamentals: KV cache sizing, prefill vs decode, roofline

Done when I can draw a request's path through nano-vllm from memory and explain every stage.

## Phase 2 · From nano-vllm to SGLang (Nov to Dec 2026)

- [ ] Read mini-sglang: radix cache, chunked prefill, overlap scheduling
- [ ] Follow the SGLang code walkthrough; run a server and `sglang.bench_serving`
- [ ] Profile one request end to end
- [ ] First SGLang PR (docs or tests first, then a bug fix)
- [ ] Attend [SGLang Summit 2026](https://www.sglang.io/summit), Nov 12–13 in San Francisco (registered)

Done when one PR is merged.

## Phase 3 · GPU programming (Dec 2026 to Jan 2027)

- [ ] GPU-Puzzles and Triton-Puzzles
- [ ] Triton softmax and matmul, with an Nsight Compute analysis of each
- [ ] Triton FlashAttention-2 forward pass

Done when I can explain from a profile why a kernel is slow.

## Phase 4 · Own engine and one area in depth (Jan to Mar 2027)

- [ ] A mini engine with continuous batching and prefix caching, in its own repo
- [ ] Pick one area: RL post-training (Miles or Tunix, training-inference alignment) or serving and scheduling
- [ ] A second upstream PR in that area

## Not now

CUDA C++ in depth, cluster scheduling and Kubernetes, large-scale training. Revisit after Phase 3.

## Reviews

- 2026-09-24: created.
- 2026-09-25: Phase 1 on track. One of two or three issues reproduced (#274, fix not started). Notes 01 to 04 not written yet, though the code behind them has been read.
- 2026-09-29: First upstream PR opened ahead of plan: nano-vllm [#280](https://github.com/GeeeekExplorer/nano-vllm/pull/280) fixes #274 and #279 (reported by me). Still one of two or three reproduced issues; notes 01 to 04 still to write.
- 2026-10-02: Reproduced nano-vllm [#170](https://github.com/GeeeekExplorer/nano-vllm/issues/170), so two issues reproduced. Added [Tunix](https://github.com/google/tunix), Google's post-training library in JAX, to the frameworks to learn.
