# {{NUMBER}} · {{TITLE}}

> Written: {{DATE}} · Status: running

<!-- Add a row for this experiment to experiments/README.md. -->

## Question

<!-- One question with a measurable answer. -->

## Setup

| | |
|---|---|
| GPU | |
| Software | <!-- from results/env.txt: torch, CUDA, project and commit --> |
| Model | |
| Workload | <!-- number of requests, input and output lengths, concurrency --> |

## Method

<!-- What you varied, what you held fixed, how many runs, warm-up. -->

## Results

| Config | Throughput (tok/s) | TTFT p50 (ms) | TTFT p95 (ms) | Decode latency (ms/token) |
|---|---|---|---|---|
| | | | | |

## Conclusion

<!-- Answer the question in two or three sentences, and explain why the numbers look this way. -->

## Caveats

<!-- What would change the result: another GPU, a larger model, longer prompts. -->

## Reproduce

```bash
bash run.sh
```
