# Papers

Status: to read · reading · read · noted (a written summary exists)

## Serving

| Paper | Idea | Status | Note |
|---|---|---|---|
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) | iteration-level scheduling, the basis of continuous batching | to read | |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) (SOSP 2023) | KV cache in fixed-size blocks, like OS paging | to read | |
| [SGLang](https://arxiv.org/abs/2312.07104) (NeurIPS 2024) | RadixAttention: prefix reuse across requests | to read | |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (OSDI 2024) | chunked prefill to balance throughput and latency | to read | |
| [DistServe](https://arxiv.org/abs/2401.09670) (OSDI 2024) | prefill and decode on separate GPUs | to read | |
| [Splitwise](https://arxiv.org/abs/2311.18677) (ISCA 2024) | phase splitting across machine types | to read | |

## Kernels

| Paper | Idea | Status | Note |
|---|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware attention with tiling | to read | |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | better parallelism and work partitioning | to read | |

## Decoding

| Paper | Idea | Status | Note |
|---|---|---|---|
| [Speculative decoding](https://arxiv.org/abs/2211.17192) (ICML 2023) | draft then verify, same output distribution | to read | |
| [EAGLE](https://arxiv.org/abs/2401.15077) (ICML 2024) | drafting at the feature level | to read | |

## RL systems

| Paper | Idea | Status | Note |
|---|---|---|---|
| [HybridFlow / verl](https://arxiv.org/abs/2409.19256) (EuroSys 2025) | flexible placement of RL dataflow on GPUs | to read | |
