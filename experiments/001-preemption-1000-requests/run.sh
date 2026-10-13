#!/usr/bin/env bash
# Reproduce this experiment end to end. Keep it runnable by someone else.
#   NANO_VLLM=/path/to/nano-vllm bash run.sh       (a checkout at bb823b3)
#   PYTHON=/path/to/python to choose the interpreter (an environment where nano-vllm imports)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
: "${NANO_VLLM:?set NANO_VLLM to a nano-vllm checkout at bb823b3}"
py="${PYTHON:-python}"

# Record the setup.
PYTHON="$py" bash ../../scripts/collect_env.sh "$NANO_VLLM" > results/env.txt

# The scheduler runs on the CPU; no GPU work happens.
PYTHONPATH="$NANO_VLLM" "$py" preempt_sim.py | tee results/output.txt

# The same workload with a larger KV cache.
for blocks in 2048 4096; do
  PYTHONPATH="$NANO_VLLM" "$py" preempt_sim.py 1000 500 300 "$blocks" > "results/output_$blocks.txt"
  tail -n 1 "results/output_$blocks.txt"
done
