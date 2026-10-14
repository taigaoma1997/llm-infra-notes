"""End-to-end run of PR #281's bench.py workload, with the time split into prefill, decode and preemption.

Workload, identical to the PR's bench.py: seed(0), 64 requests, prompts of 100-1024 random tokens,
100-1024 generated tokens each (ignore_eos), temperature 0.6, CUDA graphs on, max_model_len 4096.
Each step is timed: step() returns a positive token count for prefill and minus the batch size for decode,
and run() ends with .tolist(), which waits for the GPU, so a step's wall time is its real cost.
Each run appends one row to results/breakdown.csv. Repeat a configuration by running the command again:
it has to be a new process, because a second run in the same process would hit the prefix cache.

Usage: PYTHONPATH=<nano-vllm checkout> python bench_breakdown.py [int8] [--num-seqs N] [--no-save]
  int8 needs the PR #281 checkout; on main the quantization argument is silently ignored.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from random import randint, seed

import winfix  # noqa: F401  (Windows only; must come before nanovllm)
import nanovllm
from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler

MODEL = str(Path("~/huggingface/Qwen3-0.6B").expanduser())
BUCKETS = [(1, 8), (9, 16), (17, 32), (33, 64), (65, 512)]   # decode steps grouped by batch size

recording = False
steps = []               # (num_tokens, seconds) per step; > 0 prefill tokens, < 0 minus the decode batch size
preempted = set()
stats = dict(preemptions=0, returns=0, return_hit_tokens=0)

_step = LLMEngine.step
def timed_step(self):
    t0 = time.perf_counter()
    outputs, num_tokens = _step(self)
    if recording:
        steps.append((num_tokens, time.perf_counter() - t0))
    return outputs, num_tokens
LLMEngine.step = timed_step

_preempt = Scheduler.preempt
def counted_preempt(self, seq):
    if recording:
        stats["preemptions"] += 1
        preempted.add(seq.seq_id)
    _preempt(self, seq)
Scheduler.preempt = counted_preempt

# When a preempted sequence comes back, count the tokens it finds in its own freed blocks (prefix cache hits).
_schedule = Scheduler.schedule
def watched_schedule(self):
    seqs, is_prefill = _schedule(self)
    if recording and is_prefill:
        for seq in seqs:
            if seq.seq_id in preempted:
                preempted.discard(seq.seq_id)
                stats["returns"] += 1
                stats["return_hit_tokens"] += seq.num_cached_tokens
    return seqs, is_prefill
Scheduler.schedule = watched_schedule


def main():
    global recording
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="bf16", choices=["bf16", "int8"])
    ap.add_argument("--num-seqs", type=int, default=64)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    checkout = Path(nanovllm.__file__).resolve().parents[1].name

    seed(0)  # same order of random calls as the PR's bench.py
    prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(args.num_seqs)]
    params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(args.num_seqs)]
    prompt_tokens = sum(len(p) for p in prompts)

    llm = LLM(MODEL, enforce_eager=False, max_model_len=4096,
              quantization="int8" if args.mode == "int8" else None)
    blocks = llm.model_runner.config.num_kvcache_blocks
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)   # warmup, as in bench.py

    recording = True
    t0 = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    total = time.perf_counter() - t0
    recording = False

    pre = [(n, t) for n, t in steps if n > 0]
    dec = [(-n, t) for n, t in steps if n < 0]
    pre_s, dec_s = sum(t for _, t in pre), sum(t for _, t in dec)
    pre_tokens, dec_tokens = sum(n for n, _ in pre), sum(n for n, _ in dec)

    print(f"\n{checkout}, {args.mode}, {blocks} KV blocks, {args.num_seqs} requests")
    print(f"total {total:.2f} s (steps {pre_s + dec_s:.2f} s, other {total - pre_s - dec_s:.2f} s)")
    print(f"prefill: {len(pre)} steps, {pre_s:.2f} s, {pre_tokens} tokens "
          f"({prompt_tokens} prompt + {pre_tokens - prompt_tokens} recomputed after preemption), {pre_tokens / pre_s:.0f} tok/s")
    print(f"decode: {len(dec)} steps, {dec_s:.2f} s, {dec_tokens} tokens, {dec_s / len(dec) * 1000:.2f} ms per step")
    print(f"preemptions: {stats['preemptions']}; on return, {stats['return_hit_tokens']} tokens were found in their own freed blocks")
    row = dict(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), version=checkout, mode=args.mode,
               num_seqs=args.num_seqs, blocks=blocks, total_s=round(total, 3),
               prefill_steps=len(pre), prefill_s=round(pre_s, 3), prefill_tokens=pre_tokens,
               recompute_tokens=pre_tokens - prompt_tokens,
               decode_steps=len(dec), decode_s=round(dec_s, 3), decode_tokens=dec_tokens,
               preemptions=stats["preemptions"], returns=stats["returns"],
               return_hit_tokens=stats["return_hit_tokens"])
    print("decode by batch size:")
    for lo, hi in BUCKETS:
        sel = [t for b, t in dec if lo <= b <= hi]
        if sel:
            print(f"  batch {lo:>3}-{hi:<3}: {len(sel):>4} steps, {sum(sel):6.2f} s, {sum(sel) / len(sel) * 1000:6.2f} ms per step")
        row[f"decode_b{lo}_{hi}_steps"] = len(sel)
        row[f"decode_b{lo}_{hi}_ms_per_step"] = round(sum(sel) / len(sel) * 1000, 3) if sel else ""

    if not args.no_save:
        out = Path(__file__).resolve().parent / "results" / "breakdown.csv"
        out.parent.mkdir(exist_ok=True)
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"appended to {out}")


if __name__ == "__main__":
    main()
