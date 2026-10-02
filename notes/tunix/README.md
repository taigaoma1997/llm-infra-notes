# Tunix

> Upstream: [google/tunix](https://github.com/google/tunix) · Docs: [tunix.readthedocs.io](https://tunix.readthedocs.io/en/latest/index.html) · Status: planned

Google's LLM post-training library, written in JAX (the name stands for "Tune-in-JAX"). It covers supervised fine-tuning (full weights, LoRA, DPO, ORPO), RL (PPO, GRPO and its variants), and agentic RL with multi-turn tool use and asynchronous rollout. Models are written in Flax NNX, rollout can run on vLLM or SGLang-JAX, and the main target is TPUs.

It is the JAX and TPU counterpart to [Miles](../miles/README.md), which is PyTorch on GPUs.

Setup note: JAX has no NVIDIA GPU support on native Windows, and only experimental support under WSL2 ([JAX supported platforms](https://docs.jax.dev/en/latest/installation.html#supported-platforms)). On my laptop it would run on CPU, or on the GPU through WSL2.

## Notes

None yet.

## Resources

- [Tunix documentation](https://tunix.readthedocs.io/en/latest/index.html): [design overview](https://tunix.readthedocs.io/en/latest/design.html), [quick start](https://tunix.readthedocs.io/en/latest/quickstart.html), [algorithms](https://tunix.readthedocs.io/en/latest/algorithms.html)
- [Rollout with vLLM and SGLang-JAX](https://tunix.readthedocs.io/en/latest/rollout.html)
