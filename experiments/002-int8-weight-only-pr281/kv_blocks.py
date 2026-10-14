"""How many KV cache blocks does nano-vllm allocate, and where does the memory go?

Prints the three terms of nano-vllm's block formula, read at the start of allocate_kv_cache:
  blocks = (total * gpu_memory_utilization - used - (peak - current)) / bytes per block
  used    = total - free, as the driver sees it (weights, CUDA context, loaded kernels, PyTorch's cache, other programs)
  current = bytes held by live PyTorch tensors at that moment (mostly the weights)
  peak    = the most bytes held at once since the warmup started (weights + the largest step's temporary tensors)

Usage: PYTHONPATH=<nano-vllm checkout> python kv_blocks.py [int8]
  int8 needs the PR #281 checkout; on main the quantization argument is silently ignored.
"""
import sys
from pathlib import Path

import winfix  # noqa: F401  (Windows only; must come before nanovllm)
import torch
from nanovllm import LLM
from nanovllm.engine.model_runner import ModelRunner

MODEL = str(Path("~/huggingface/Qwen3-0.6B").expanduser())
MiB = 2**20

_allocate_kv_cache = ModelRunner.allocate_kv_cache
def logged_allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    print(f"total={total / MiB:.0f}  used={(total - free) / MiB:.0f}  peak={stats['allocated_bytes.all.peak'] / MiB:.0f}  "
          f"current={stats['allocated_bytes.all.current'] / MiB:.0f}  reserved={torch.cuda.memory_reserved() / MiB:.0f} MiB")
    _allocate_kv_cache(self)
ModelRunner.allocate_kv_cache = logged_allocate_kv_cache


def main():
    mode = "int8" if "int8" in sys.argv[1:] else "bf16"
    # enforce_eager skips recording CUDA graphs, which happens after the block count is fixed
    llm = LLM(MODEL, enforce_eager=True, quantization="int8" if mode == "int8" else None)
    checkout = Path(sys.modules["nanovllm"].__file__).resolve().parents[1].name
    print(f"{checkout} {mode}: {llm.model_runner.config.num_kvcache_blocks} blocks")


if __name__ == "__main__":
    main()
