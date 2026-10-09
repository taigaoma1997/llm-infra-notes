# triton-windows

> Upstream: [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows) · Status: one docs PR, merged

Triton's official wheels are Linux only, so on Windows `torch.compile` needs this fork to generate and compile GPU kernels. I use it on my laptop to run nano-vllm, whose small layers (RMSNorm, SiLU, rotary embedding, sampler) are compiled with `torch.compile`. This is not a study track: the folder holds what I contributed back after hitting a bug.

The repo's default branch is `readme`, which holds documentation and is rebased from time to time; code PRs go to `main-windows` ([BUILD.md](https://github.com/triton-lang/triton-windows/blob/readme/BUILD.md#branches-in-this-repo)).

## Contributions

| PR | What | Status | Write-up |
|---|---|---|---|
| [#56](https://github.com/triton-lang/triton-windows/pull/56) | Document a PyTorch 2.6 `torch.compile` error on Windows (`PermissionError: [WinError 5]` in `TritonBundler.read_and_emit`) in the README's Known issues | merged 2026-10-09 | [56](issues/56-pytorch-2.6-os-replace.md) |
