# Miles and RL post-training

> Upstream: [radixark/miles](https://github.com/radixark/miles) · Status: planned

RL post-training alternates generation (rollout) and training on the same model. The system problems are keeping both sides busy, moving weights from trainer to inference engine quickly, and making sure the two sides compute the same numbers. That last one, training-inference alignment, is close to the numerical debugging I already do as an MLE.

## Plan

1. Run a small GRPO job and measure how time splits between rollout, training and weight sync.
2. Read how Miles keeps training and inference consistent, including MoE routing.
3. Compare with slime (which Miles was forked from) and verl.

## Notes

| # | Question | Status |
|---|---|---|
| 01 | Where does the time go in one RL step? | planned |
| 02 | Why do training and inference logprobs differ, and how does Miles close the gap? | planned |

## Resources

- [Introducing Miles (LMSYS)](https://www.lmsys.org/blog/2025-11-19-miles/)
- [slime](https://github.com/THUDM/slime) and its [documentation](https://thudm.github.io/slime/)
- [verl](https://github.com/volcengine/verl) and the [HybridFlow paper](https://arxiv.org/abs/2409.19256)
- [Hands-on Modern RL, appendix B.1: RL training systems](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/rl-infrastructure)
