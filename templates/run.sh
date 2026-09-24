#!/usr/bin/env bash
# Reproduce this experiment end to end. Keep it runnable by someone else.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

# Record the setup. Pass the path of the upstream checkout you benchmarked.
bash ../../scripts/collect_env.sh "${UPSTREAM:-}" > results/env.txt

# TODO: the command(s) that produce the numbers in README.md, for example:
# python bench.py --model Qwen/Qwen3-0.6B --batch-sizes 1 8 32 | tee results/summary.txt
