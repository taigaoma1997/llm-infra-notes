# LLM Infra Notes

I'm an ML engineer working on ads ranking models, moving into LLM inference and training systems.
This repo is my public learning log: source-code notes, reproducible experiments, and upstream contributions.

**Now:** reading nano-vllm and reproducing its open issues.
**Next:** mini-sglang, then a first contribution to SGLang.

## Highlights

<!-- Three rows at most, and only things that exist. Delete this section until you have the first one. -->

| | |
|---|---|
| Note | [How nano-vllm chooses between prefill and decode](notes/nano-vllm/) |
| Issue | [Reproduced nano-vllm #NNN: short title](notes/nano-vllm/issues/) |
| Experiment | [CUDA Graph on vs off: decode latency](experiments/) |

## Progress

| Track | Status | Where |
|---|---|---|
| Inference fundamentals | in progress | [notes/fundamentals](notes/fundamentals/) |
| nano-vllm | in progress | [notes/nano-vllm](notes/nano-vllm/) |
| mini-sglang and SGLang | next | [notes/sglang](notes/sglang/) |
| vLLM | planned | [notes/vllm](notes/vllm/) |
| GPU programming (Triton, CUDA) | planned | [notes/gpu](notes/gpu/) |
| RL post-training (Miles) | planned | [notes/miles](notes/miles/) |
| Papers | ongoing | [notes/papers](notes/papers/) |

Full plan: [ROADMAP.md](ROADMAP.md) · Issues and PRs: [CONTRIBUTIONS.md](CONTRIBUTIONS.md) · Experiments: [experiments/](experiments/)

## Recent updates

<!-- Newest first, one line each, linked to the weekly update. Keep the last five. -->

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

Open an issue on this repo, or email <your personal email>.
