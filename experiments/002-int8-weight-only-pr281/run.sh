#!/usr/bin/env bash
# Reproduce this experiment end to end. Keep it runnable by someone else.
#   NANO_VLLM_MAIN=<nano-vllm at bb823b3> NANO_VLLM_PR=<PR #281 at c3b7110> PYTHON=<python> bash run.sh
# Get the PR with: git fetch origin pull/281/head:pr-281 && git worktree add ../pr-281 pr-281
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
: "${NANO_VLLM_MAIN:?set NANO_VLLM_MAIN to a nano-vllm checkout at bb823b3}"
: "${NANO_VLLM_PR:?set NANO_VLLM_PR to a checkout of PR #281}"
py="${PYTHON:-python}"

PYTHON="$py" bash ../../scripts/collect_env.sh "$NANO_VLLM_MAIN" "$NANO_VLLM_PR" > results/env.txt

# 1. KV cache blocks and the memory terms behind them
{ PYTHONPATH="$NANO_VLLM_MAIN" "$py" kv_blocks.py
  PYTHONPATH="$NANO_VLLM_PR" "$py" kv_blocks.py
  PYTHONPATH="$NANO_VLLM_PR" "$py" kv_blocks.py int8; } | tee results/kv_blocks.txt

# 2. The kernel alone
PYTHONPATH="$NANO_VLLM_PR" "$py" bench_kernel.py | tee results/kernel_run1.txt
PYTHONPATH="$NANO_VLLM_PR" "$py" bench_kernel.py --m 2000 2017 2048 4000 4096 \
  --out results/kernel_m2000.csv | tee results/kernel_run2.txt

# 3. End to end, three rounds, alternating the configurations
for i in 1 2 3; do
  PYTHONPATH="$NANO_VLLM_MAIN" "$py" bench_breakdown.py
  PYTHONPATH="$NANO_VLLM_PR" "$py" bench_breakdown.py
  PYTHONPATH="$NANO_VLLM_PR" "$py" bench_breakdown.py int8
done

# 4. Hardware numbers, copy bandwidth, and what the kernel compiles to
PYTHONPATH="$NANO_VLLM_PR" "$py" gpu_facts.py | tee results/gpu_facts.txt

# 5. The size of every prefill step (scheduler only, CPU)
PYTHONPATH="$NANO_VLLM_MAIN" "$py" prefill_steps.py | tee results/prefill_steps.txt
