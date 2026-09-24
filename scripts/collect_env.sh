#!/usr/bin/env bash
# Print the setup behind a result, for experiment and issue write-ups.
#   scripts/collect_env.sh [path to an upstream checkout ...]
# Missing tools are reported, not fatal.
set -uo pipefail

echo "date:     $(date +%Y-%m-%d)"
echo "os:       $(uname -srm)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | sed 's/^/gpu:      /'
else
  echo "gpu:      (nvidia-smi not found)"
fi

py="$(command -v python3 || command -v python || true)"
if [ -n "$py" ]; then
  "$py" - <<'EOF'
import platform
from importlib import metadata

print(f"python:   {platform.python_version()}")
try:
    import torch
    print(f"torch:    {torch.__version__} (CUDA {torch.version.cuda})")
except Exception:
    print("torch:    (not installed)")
for dist in ("triton", "flash-attn", "transformers", "nano-vllm", "sglang", "vllm"):
    try:
        print(f"{dist + ':':<10}{metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        pass
EOF
fi

for repo in "$@"; do
  [ -n "$repo" ] || continue
  if git -C "$repo" rev-parse --short HEAD >/dev/null 2>&1; then
    name="$(basename "$(cd "$repo" && pwd)")"
    commit="$(git -C "$repo" rev-parse --short HEAD)"
    dirty=""
    git -C "$repo" diff --quiet || dirty=" (uncommitted changes)"
    echo "repo:     $name @ $commit$dirty"
  else
    echo "repo:     $repo (not a git checkout)"
  fi
done
