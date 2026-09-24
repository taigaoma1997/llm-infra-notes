# Experiments

Each folder answers one question with numbers. Every result lists the hardware, the versions, and the command that produced it. Create a new one with `scripts/new.sh exp <slug> "<question>"`.

| # | Question | Answer in one line | Status |
|---|---|---|---|
| | | | |

## Backlog

- nano-vllm: how much does CUDA Graph save per decode step? Compare `enforce_eager=True` with the default at batch sizes 1, 8 and 32.
- nano-vllm: how does throughput change as `max_num_seqs` grows, where does it level off, and why?
- nano-vllm: with a shared 2K-token system prompt, how much prefill time does prefix caching save?
- KV cache: does measured memory match the formula in [fundamentals](../notes/fundamentals/) for Qwen3-0.6B at several context lengths?
