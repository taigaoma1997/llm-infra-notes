# LLM Infra Notes

This repo is my public learning log: source-code notes, reproducible experiments, and upstream contributions.

**Now:** reading nano-vllm and reproducing its open issues.
**Next:** mini-sglang, then a first contribution to SGLang.

## Highlights

<!-- Three rows at most, and only things that exist. Delete this section until you have the first one. -->

| | |
|---|---|
| PR | [nano-vllm #280: finish or reject sequences that cannot fit in the KV cache](https://github.com/GeeeekExplorer/nano-vllm/pull/280) (open) |
| Issue | [Reproduced nano-vllm #274 and reported #279: the engine crashes when a request cannot fit in the KV cache](notes/nano-vllm/issues/274-kv-cache-exhausted-assert.md) |

## Progress

| Track | Status | Where |
|---|---|---|
| Inference fundamentals | in progress | [notes/fundamentals](notes/fundamentals/) |
| nano-vllm | in progress | [notes/nano-vllm](notes/nano-vllm/) |
| mini-sglang and SGLang | next | [notes/sglang](notes/sglang/) |
| vLLM | planned | [notes/vllm](notes/vllm/) |
| GPU programming (Triton, CUDA) | planned | [notes/gpu](notes/gpu/) |
| RL post-training (Miles) | planned | [notes/miles](notes/miles/) |
| Post-training in JAX (Tunix) | planned | [notes/tunix](notes/tunix/) |
| Papers | ongoing | [notes/papers](notes/papers/) |

Full plan: [ROADMAP.md](ROADMAP.md) · Issues and PRs: [CONTRIBUTIONS.md](CONTRIBUTIONS.md) · Experiments: [experiments/](experiments/)

## Recent updates

<!-- Newest first, one line each, linked to the weekly update. Keep the last five. -->

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
- Each number comes with the hardware, the versions, and the command that produced it.
- Notes marked `draft` may be wrong. Corrections are welcome as issues.
- Everything here is based on public code and public material only.

## Contact

Open an issue on this repo, or email taigaom@umich.edu.
