# Experiments

Each folder answers one question with numbers. Every result lists the hardware, the versions, and the command that produced it. Create a new one with `scripts/new.sh exp <slug> "<question>"`.

| # | Question | Answer in one line | Status |
|---|---|---|---|
| [001](001-preemption-1000-requests/) | When does nano-vllm preempt, with 1000 requests and a full KV cache? | With 1024 blocks: 16 prefill steps fill the cache, 171 sequences are preempted at step 29 and 85 at step 285, and preemption adds 68% more prefill work. With 2048 blocks, sequences beyond the first 512 hold blocks without decoding | done |
