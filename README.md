# LLM Infra Notes

This repo is my public learning log: source-code notes, reproducible experiments, and upstream contributions.

- **Now:** reproducing nano-vllm's open issues, and reading mini-sglang.
- **Next:** a first contribution to SGLang.
- **Upcoming:** [SGLang Summit 2026](https://www.sglang.io/summit), Nov 12–13 in San Francisco (registered).

## Highlights

<!-- One row per PR or issue, only things that exist. When an "In progress" row is done, relabel it "Issue". -->

| | |
|---|---|
| PR | [triton-windows #56: document a PyTorch 2.6 `torch.compile` error on Windows](https://github.com/triton-lang/triton-windows/pull/56) (open). Found while running nano-vllm, reproduced without it, traced to a PyTorch bug that was fixed in 2.7 but never reported: [write-up](notes/triton-windows/issues/56-pytorch-2.6-os-replace.md) |
| PR | [nano-vllm #280: finish or reject sequences that cannot fit in the KV cache](https://github.com/GeeeekExplorer/nano-vllm/pull/280) (open) |
| Issue | [nano-vllm #274](notes/nano-vllm/issues/274-kv-cache-exhausted-assert.md): reproduced the crash when a request cannot fit in the KV cache, and reported a second path to it as #279 |
| Issue | [nano-vllm #170](notes/nano-vllm/issues/170-rmsnorm-fp32-residual.md): found why RMSNorm corrupts the residual in fp32, plus a second buggy path the issue missed. Compared my fix with the three open PRs; [#171](https://github.com/GeeeekExplorer/nano-vllm/pull/171) already fixes both, so no PR from me |
| Issue | [nano-vllm #190](notes/nano-vllm/issues/190-cuda-graph-block-tables.md): CUDA graph replay fails once a sequence grows past `max_model_len`. Reproduced on upstream, showed that the fix the issue suggests only moves the crash, and tested the two open PRs on the same script: [#270](https://github.com/GeeeekExplorer/nano-vllm/pull/270) fixes the root cause, so no PR from me so far |

## Progress

| Track | Status | Where |
|---|---|---|
| Inference fundamentals | in progress | [notes/fundamentals](notes/fundamentals/) |
| nano-vllm | in progress | [notes/nano-vllm](notes/nano-vllm/) |
| mini-sglang | in progress | [notes/mini-sglang](notes/mini-sglang/) |
| SGLang | next | [notes/sglang](notes/sglang/) |
| vLLM | planned | [notes/vllm](notes/vllm/) |
| GPU programming (Triton, CUDA) | planned | [notes/gpu](notes/gpu/) |
| RL post-training (Miles) | planned | [notes/miles](notes/miles/) |
| Post-training in JAX (Tunix) | planned | [notes/tunix](notes/tunix/) |
| Papers | ongoing | [notes/papers](notes/papers/) |

Full plan: [ROADMAP.md](ROADMAP.md) · Issues and PRs: [CONTRIBUTIONS.md](CONTRIBUTIONS.md) · Experiments: [experiments/](experiments/)

## Recent updates

<!-- Newest first, one line each, linked to the weekly update. Keep the last five. -->

- **2026-10-05** Started mini-sglang: set it up in WSL2 on Windows and ran it end to end. Looked inside nano-vllm's recorded CUDA graphs for #190, reproduced it, and tested the open PRs that fix it. Opened a docs PR to triton-windows (#56): [update](updates/2026/2026-10-05.md)
- **2026-09-29** Fixed nano-vllm #274, reported #279, opened my first upstream PR (#280), reproduced #170 and compared its fixes, started #190: [update](updates/2026/2026-09-29.md)
- **2026-09-25** Sorted nano-vllm's open issues and reproduced #274: [update](updates/2026/2026-09-25.md)
- **2026-09-24** Started this repo and wrote down what I did before it: [update](updates/2026/2026-09-24-backfill.md)

## Layout

```
notes/         code and concept notes, one folder per project or topic
experiments/   one folder per question, with the script and the numbers
updates/       weekly updates
templates/     templates for notes, issue write-ups, experiments, updates
scripts/       new.sh creates files from templates; collect_env.sh records the setup
```

## Conventions

- Each note answers one question. Code references are GitHub permalinks pinned to a commit.
- Each project folder can also have Q&A pages (`QA-<topic>.md`): the questions I asked while learning a topic, with answers and self-checks. They are in Chinese, my working language for studying; everything else is in English.
- Each number comes with the hardware, the versions, and the command that produced it.
- Notes marked `draft` may be wrong. Corrections are welcome as issues.
- Everything here is based on public code and public material only.

## Contact

Open an issue on this repo, or email taigaom@umich.edu.
