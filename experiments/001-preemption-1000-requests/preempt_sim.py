"""Drive nano-vllm's real Scheduler and BlockManager with a fake model, and print what each step does.

1000 requests arrive at once. Each prompt has 500 tokens (distinct per request, so no prefix sharing)
and generates exactly 300 tokens (ignore_eos). The KV cache has 1024 blocks of 256 tokens.
The model is replaced by "every scheduled sequence gets token 1", so this runs on the CPU in seconds.

Usage: PYTHONPATH=<nano-vllm checkout at bb823b3> python preempt_sim.py [requests prompt_len max_tokens blocks]
"""
import sys
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

N, PROMPT, MAX_TOKENS, BLOCKS = 1000, 500, 300, 1024
if len(sys.argv) > 1:
    N, PROMPT, MAX_TOKENS, BLOCKS = map(int, sys.argv[1:5])

# The fields Scheduler and BlockManager read from Config; the defaults of nano-vllm except the block count.
Sequence.block_size = 256
config = SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=16384, eos=-1,
                         kvcache_block_size=256, num_kvcache_blocks=BLOCKS)
sch = Scheduler(config)
for i in range(N):
    sch.add(Sequence([1000 + i] * PROMPT, SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True)))

preempted = []
original_preempt = sch.preempt
def counting_preempt(seq):
    preempted.append(seq.seq_id)
    original_preempt(seq)
sch.preempt = counting_preempt

rows = []
step = 0
while not sch.is_finished():
    step += 1
    preempted.clear()
    seqs, is_prefill = sch.schedule()
    free_after_schedule = len(sch.block_manager.free_block_ids)
    tokens = sum(s.num_scheduled_tokens for s in seqs)
    recompute = sum(1 for s in seqs if is_prefill and s.num_tokens > PROMPT)  # back after preemption
    lens = sorted({len(s) for s in seqs}) if not is_prefill else []
    sch.postprocess(seqs, [1] * len(seqs), is_prefill)
    finished = sum(1 for s in seqs if s.is_finished)
    rows.append(dict(step=step, kind="prefill" if is_prefill else "decode", n=len(seqs), tokens=tokens,
                     pre=len(preempted), fin=finished, free=free_after_schedule, running=len(sch.running),
                     waiting=len(sch.waiting), recompute=recompute,
                     lens=(lens[0], lens[-1]) if lens else None))

# Merge consecutive steps that look alike into one line.
def key(r):
    return (r["kind"], r["n"], r["pre"] > 0, r["fin"] > 0, r["recompute"] > 0)
groups = []
for r in rows:
    if groups and key(groups[-1][-1]) == key(r) and not r["pre"] and not r["fin"]:
        groups[-1].append(r)
    else:
        groups.append([r])

print(f"{N} requests x {PROMPT} tokens, max_tokens {MAX_TOKENS}, {BLOCKS} blocks; total steps {step}")
print("free = free blocks after schedule(); running and waiting = after postprocess()")
print("steps        kind     seqs/step  tokens/step  preempt  finish  recompute  free  running  waiting  seq len (decode)")
for g in groups:
    a, b = g[0], g[-1]
    rng = f"{a['step']}" if a is b else f"{a['step']}-{b['step']}"
    pre = sum(r["pre"] for r in g)
    fin = sum(r["fin"] for r in g)
    rec = sum(r["recompute"] for r in g)
    lens = f"{a['lens'][0]}..{b['lens'][1]}" if a["lens"] else ""
    print(f"{rng:<12} {a['kind']:<8} {a['n']:>9}  {a['tokens']:>11}  {pre:>7}  {fin:>6}  {rec:>9}  "
          f"{b['free']:>4}  {b['running']:>7}  {b['waiting']:>7}  {lens}")
print("(tokens/step is the first step of a merged line)")

prefill_rows = [r for r in rows if r["kind"] == "prefill"]
print(f"summary: {step} steps ({len(prefill_rows)} prefill, {step - len(prefill_rows)} decode), "
      f"{sum(r['pre'] for r in rows)} preemptions, "
      f"{sum(r['tokens'] for r in prefill_rows)} prefill tokens (prompts alone: {N * PROMPT}), "
      f"at most {max(r['running'] for r in rows)} sequences in running")
