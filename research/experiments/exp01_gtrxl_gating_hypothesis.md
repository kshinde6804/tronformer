# Experiment 01: GTrXL GRU Gating

**Date:** 2026-06-16  
**Branch:** exp/gtrxl-gating  
**Phase:** 1 (recurrent block only)

## What is changing

Replace `nn.TransformerEncoder` (standard residual connections) with a custom
`GatedTransformerEncoder` that uses GRU-type gating at each sublayer:

```
Standard:  x = x + Attention(LN(x))      # residual addition
           x = x + FFN(LN(x))

GTrXL:     x = gate(x, Attention(LN(x))) # GRU gate replaces +
           x = gate(x, FFN(LN(x)))
```

GRU gate initialized with bias `bg=2.0` so `z ≈ 0.12` at init → identity-dominant
(near-Markovian policy at t=0, matches what `norm_first=True` was already aiming for).

All other components unchanged: `InputProjection`, `FactoredDuelingHeads`, sinusoidal
PE, causal mask, training recipe, env parameters.

## Why

- Our current Transformer (pre-norm, no gating) trails the LSTM by ~5% over ZI.
- GTrXL (Parisotto et al. 2020) consistently closes the Transformer-vs-LSTM gap across
  RL benchmarks. The mechanism: GRU gating provides adaptive memory management that pure
  residual addition lacks.
- The identity initialization gives a stable starting point identical to the LSTM's
  internal gating advantage.
- Pre-norm (`norm_first=True`) is already in place — adding gating completes the GTrXL recipe.

## Expected effect on % over ZI

**Current best Transformer:** 9.7% (checkpoint: tron_env_c_xfmr_best.pt)  
**Expected after:** 12–15% over ZI  
**Confidence:** medium-high (GTrXL reliably improves over standard Transformer in RL;
uncertainty is whether TRON's 64-step context is long enough to benefit from gating).

## Failure modes

1. No improvement: gating overhead not needed at max_seq=64 (short enough that standard
   residual connections are stable). Then try recipe changes (exp02).
2. Divergence: gate initialization wrong. Check bg parameter — increase to 4.0 if unstable.
3. Marginal gain (<1%): architecture improvement is real but small. Count as non-improving
   iteration toward Phase 2 trigger.
