# Experiment 03: Longer Context Window (max_seq=128)

**Date:** 2026-06-17  
**Branch:** exp/max-seq-128  
**Phase:** 1 (training recipe / hyperparameter)

## What is changing

Increase `max_seq` from 64 to 128 — the Transformer context window, i.e. how many past
arrival observations the agent can attend over during inference.

No code changes. Training command adds `--xfmr-max-seq 128`.
Architecture applied: current best (GTrXL if exp01+exp02 kept, else whatever is current best).

## Why

Each TRON episode has T=2000 arrivals. At max_seq=64 the agent only sees the last 64 events.
Market microstructure features like trending fundamentals, accumulated position, and volume
imbalance have meaningful autocorrelation at lags beyond 64. The LSTM in `TRONNet` has
unbounded memory (albeit compressed); the Transformer is hard-capped.

Doubling the context to 128 gives the Transformer more past arrivals to attend over, potentially
capturing medium-term trends and more stable position tracking.

Compute cost: self-attention is O(T²) — doubling seq length doubles attention FLOPs but the
forward pass is dominated by the FFN and the per-step environment cost. Expected: ~15–25% slower
training (CPU-bound env dominates).

## Expected effect on % over ZI

**Expected:** +1 to +3% over current best  
**Confidence:** medium (benefit depends on whether market signal has structure at lags > 64)

## Results (2026-06-18)

```
ZI   mean PnL = +1253.59  sd=2423.76  sem=24.24
TRON mean PnL = +1455.96  sd=2503.66  sem=25.04
delta         = +202.37  sem=34.85  z=5.81
             = +16.1% over ZI
```

**Decision: DISCARD.** Result is identical to exp01 (GTrXL, max_seq=64) — same weights,
same eval numbers. Root cause: TRON receives only ~24 arrivals/episode (lam=0.012,
T=2000 → ~24 TRON arrivals). Context window is never truncated at max_seq=64, so
max_seq=128 is architecturally identical for this dataset. The exp03 training reproduced
exp01 exactly (seed=0, functionally equivalent architecture). Consecutive non-improving: 2.

**Key finding:** Any max_seq ≥ 30 is equivalent for Env C. Future context-length
experiments should focus on other improvements or ensure context IS the bottleneck.

## Failure modes

1. No improvement: relevant market state is captured within the last 64 steps. Then longer context
   just adds noise the attention has to learn to ignore.
2. Slower training (known): ~20% more time per run. Accept this.

## Implementation

Zero code change. Train with:
```bash
--xfmr-max-seq 128
```
Applied to current-best arch.
