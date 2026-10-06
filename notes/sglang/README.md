# SGLang

> Upstream: [sgl-project/sglang](https://github.com/sgl-project/sglang) · Status: next, after mini-sglang

## Plan

1. Read mini-sglang first: radix cache, chunked prefill, overlap scheduling, tensor parallelism, each compared with nano-vllm. Notes: [notes/mini-sglang](../mini-sglang/).
2. Follow a request through SGLang: HTTP server → TokenizerManager → Scheduler → model runner → DetokenizerManager.
3. Run a server locally, measure it with `python -m sglang.bench_serving`, and profile one request.
4. First PR: documentation or tests to learn the workflow, then a bug fix.

## Notes

The mini-sglang notes come first: [notes/mini-sglang](../mini-sglang/).

## Events

- [SGLang Summit 2026](https://www.sglang.io/summit), Nov 12–13, 2026, Fort Mason Center, San Francisco, hosted by LMSYS. Registered on 2026-10-04. The detailed agenda is not out yet.

## Resources

- [SGLang code walkthrough (Chinese)](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)
- [Contribution guide](https://docs.sglang.ai/developer_guide/contribution_guide.html)
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang): tutorial from basics to a first PR
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): Lecture 35 covers SGLang performance work
- [SGLang paper](https://arxiv.org/abs/2312.07104)
