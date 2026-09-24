# Notes

One folder per project or topic. Each note answers one question, and each folder's README lists its notes in reading order.

| Folder | What's inside | Status |
|---|---|---|
| [fundamentals](fundamentals/) | KV cache sizing, prefill vs decode, roofline, GPU memory | in progress |
| [nano-vllm](nano-vllm/) | the smallest complete engine: scheduler, block manager, model runner | in progress |
| [sglang](sglang/) | mini-sglang first, then SGLang itself | next |
| [vllm](vllm/) | the engine PagedAttention came from, compared with nano-vllm | planned |
| [gpu](gpu/) | Triton and CUDA exercises, profiling | planned |
| [miles](miles/) | RL post-training infrastructure | planned |
| [papers](papers/) | reading list with short notes | ongoing |

## Suggested order for the inference track

1. fundamentals: know what limits a decode step before reading any engine
2. nano-vllm: every core idea in a codebase small enough to read in a week
3. mini-sglang: adds radix cache, chunked prefill, overlap scheduling
4. SGLang: the production engine, where contributions happen
