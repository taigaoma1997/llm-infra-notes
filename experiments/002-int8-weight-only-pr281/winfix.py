"""Let nano-vllm start on Windows without changing its code (upstream #261). Import this before nanovllm.

PyTorch on Windows has no NCCL and no libuv; single-GPU runs need neither. Also turns off the Triton
bundle in torch.compile's cache, which fails on Windows with torch 2.6 (triton-windows #56). No-op on Linux.
"""
import os
import platform

os.environ.setdefault("USE_LIBUV", "0")
if platform.system() == "Windows":
    os.environ.setdefault("TORCHINDUCTOR_BUNDLE_TRITON_INTO_FX_GRAPH_CACHE", "0")
    import torch.distributed as dist

    _init_process_group = dist.init_process_group

    def _init_with_gloo(backend=None, *args, **kwargs):
        return _init_process_group("gloo" if backend == "nccl" else backend, *args, **kwargs)

    dist.init_process_group = _init_with_gloo
