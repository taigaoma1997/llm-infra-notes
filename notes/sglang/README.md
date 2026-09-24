# SGLang

> Upstream: [sgl-project/sglang](https://github.com/sgl-project/sglang) · Status: next, after nano-vllm

## Plan

1. Read [mini-sglang](https://github.com/sgl-project/mini-sglang): radix cache, chunked prefill, overlap scheduling, tensor parallelism. Compare each with nano-vllm.
2. Follow a request through SGLang: HTTP server → TokenizerManager → Scheduler → model runner → DetokenizerManager.
3. Run a server locally, measure it with `python -m sglang.bench_serving`, and profile one request.
4. First PR: documentation or tests to learn the workflow, then a bug fix.

## Notes

| # | Question | Status |
|---|---|---|
| 01 | What does mini-sglang add on top of nano-vllm, and why? | planned |
| 02 | How does RadixAttention find and reuse a shared prefix? | planned |
| 03 | How does overlap scheduling hide CPU work behind GPU work? | planned |

## Resources

- [Mini-SGLang announcement (LMSYS)](https://www.lmsys.org/blog/2025-12-17-minisgl/)
- [SGLang code walkthrough (Chinese)](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)
- [Contribution guide](https://docs.sglang.ai/developer_guide/contribution_guide.html)
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang): tutorial from basics to a first PR
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): Lecture 35 covers SGLang performance work
- [SGLang paper](https://arxiv.org/abs/2312.07104)
