"""Replay the bench workload through nano-vllm's real scheduler (no model, CPU only) and list every prefill step.

Scheduling depends only on token counts, so with the same block count this reproduces the real run step for step:
163 blocks is what main (bf16) gets on the 8 GB laptop GPU, 180 what PR #281 with int8 gets.
Then estimates the linear-layer time of all prefill steps from the 28-layer totals measured by bench_kernel.py
(results/kernel_run1.txt and kernel_run2.txt), interpolating between the measured M.

Usage: PYTHONPATH=<nano-vllm checkout at bb823b3> python prefill_steps.py
"""
from random import randint, seed
from types import SimpleNamespace

import numpy as np
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

# Linear layers of all 28 layers per step, ms, from bench_kernel.py on the RTX 4060 Laptop GPU
M_TAB = [1, 8, 16, 32, 64, 128, 256, 1024, 2000, 2017, 2048, 4000, 4096, 16384]
BF16 = [7.77, 7.48, 7.86, 7.91, 8.52, 9.86, 16.49, 53.00, 109.60, 108.22, 107.46, 210.68, 211.96, 858.10]
INT8 = [5.88, 6.26, 6.25, 6.35, 7.08, 9.29, 16.40, 57.98, 125.53, 122.69, 123.83, 243.57, 253.73, 1770.51]

seed(0)  # same workload as bench_breakdown.py
prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(64)]
max_tokens = [randint(100, 1024) for _ in range(64)]

for blocks, label in ((163, "main, bf16"), (180, "PR #281, int8")):
    Sequence.block_size = 256
    config = SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=16384, eos=-1,
                             kvcache_block_size=256, num_kvcache_blocks=blocks)
    sch = Scheduler(config)
    for p, mt in zip(prompts, max_tokens):
        sch.add(Sequence(p, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)))
    preemptions = [0]
    original_preempt = sch.preempt
    def counting_preempt(seq, original_preempt=original_preempt):
        preemptions[0] += 1
        original_preempt(seq)
    sch.preempt = counting_preempt

    prefill_m, decode_steps = [], 0
    while not sch.is_finished():
        seqs, is_prefill = sch.schedule()
        if is_prefill:
            prefill_m.append(sum(s.num_scheduled_tokens for s in seqs))
        else:
            decode_steps += 1
        sch.postprocess(seqs, [1] * len(seqs), is_prefill)

    big = [m for m in prefill_m if m >= 8192]
    print(f"\n{label}: {blocks} blocks -> {len(prefill_m)} prefill steps, {decode_steps} decode steps, "
          f"{preemptions[0]} preemptions, {sum(prefill_m)} prefill tokens")
    print(f"  prefill tokens per step: {prefill_m}")
    print(f"  {len(big)} steps of 8192 tokens or more ({sum(big)} tokens); {len(prefill_m) - len(big)} smaller steps "
          f"({sum(prefill_m) - sum(big)} tokens)")
    for name, tab in (("bf16", BF16), ("int8", INT8)):
        est = sum(float(np.interp(m, M_TAB, tab)) for m in prefill_m) / 1000
        print(f"  linear layers over these prefill steps with the {name} kernel: about {est:.2f} s")
