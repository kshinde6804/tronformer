# Experiment 02: Context-Encoder-Specific Learning Rate

**Date:** 2026-06-17  
**Branch:** exp/encoder-lr  
**Phase:** 1 (training recipe change)

## What is changing

Split the Adam optimizer into two parameter groups with different learning rates:

```python
head_params = set(id(p) for p in q_net.heads.parameters())
encoder_params = [p for p in q_net.parameters() if id(p) not in head_params]
self.optimizer = optim.Adam([
    {'params': encoder_params, 'lr': cfg.lr * cfg.encoder_lr_scale},  # 1e-5
    {'params': heads_params,   'lr': cfg.lr},                          # 1e-4
])
```

New `TrainerConfig` field: `encoder_lr_scale: float = 1.0` (default = 1.0 → no change).
New CLI arg: `--encoder-lr-scale 0.1`.

Applied to the current best architecture (GTrXL if exp01 kept, else standard Transformer).
All other training recipe params unchanged.

## Why

Luo et al. (NeurIPS 2024) "Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific
Learning Rate" — directly applicable to our setup:
- Recurrent: ✓ (Transformer with episode-level replay)
- Off-policy: ✓ (R2D2-style replay buffer)
- Context encoder: `input_proj` + Transformer stack

Their finding: uniform LR causes the encoder to overfit to non-stationary TD targets from the
replay buffer. A 10× lower encoder LR stabilizes encoder representations and improves policy
evaluation. Average improvement across their benchmarks: +1–4% over uniform-LR baseline.

Our encoder (input_proj + Transformer) faces exactly this non-stationarity: the value function
target changes as the policy improves, but the encoder is being updated to compress past arrivals
into a useful representation. A lower encoder LR lets representations stabilize between policy
updates.

## Expected effect on % over ZI

**Baseline (from exp01 decision):** current best transformer % over ZI  
**Expected after:** +1 to +4% additional  
**Confidence:** high (directly applicable mechanism, NeurIPS-validated finding)

## Implementation details

1. `TrainerConfig`: add `encoder_lr_scale: float = 1.0`
2. `TRONTrainer.__init__`: when `encoder_lr_scale != 1.0`, create two Adam param groups
3. `train.py`: add `--encoder-lr-scale` CLI arg (forwarded to TrainerConfig)
4. The scale is saved in `cfg.__dict__` → checkpoint is self-describing

## Failure modes

1. No improvement: encoder representations are stable enough at the full LR in our short-horizon
   (max_seq=64) setting. Then the non-stationarity problem is less severe than at longer context.
2. Degradation: LR too low for encoder to learn the right features in 100k episodes. Try
   `encoder_lr_scale=0.3` as a follow-up if this happens.

## References

- Luo et al. NeurIPS 2024 "Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific
  Learning Rate" https://arxiv.org/abs/2407.16053
  GitHub: https://github.com/FanmingL/Recurrent-Offpolicy-RL
