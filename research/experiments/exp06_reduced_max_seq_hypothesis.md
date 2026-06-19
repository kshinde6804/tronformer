# exp06: Reduced max_seq=24

## What is changing
Reduce Transformer context window from 64 to 24 steps (`--xfmr-max-seq 24`). All other
hyperparameters unchanged. No code changes required.

Note: max_seq=32 was originally proposed in checkpoint_1, but analysis of Poisson arrival
distribution (lam=0.012, T=2000 → mean=24 arrivals, σ=4.9) shows:
  - max_seq=64: 0% truncation (confirmed by exp03)
  - max_seq=32: ~5% truncation (nearly a null, similar to exp03)
  - max_seq=24: ~45% truncation (meaningful regularization/context test)

Updated to max_seq=24 to produce a real experimental signal.

## Why
Two hypotheses:
1. **Regularization**: Forcing the model to work with truncated context in ~45% of episodes
   may prevent over-reliance on full episode context and improve generalization.
2. **Positional encoding quality**: With max_seq=64 but only ~24 arrivals, most of the
   sinusoidal PE table is never used. Aligning max_seq to the actual context length may
   improve the relative quality of position representations used.

Expected effect is ambiguous: truncation may hurt by removing useful context in long
episodes, or may help as a regularizer. Low expected standalone impact.

## Expected effect
-1% to +1% over ZI. Low confidence. This is primarily a diagnostic: if the result is
significantly worse, it confirms that context beyond 24 steps is useful; if neutral or
better, it suggests the model doesn't rely on full context.

## Previous best
Checkpoint: `out_exp_gtrxl_best.pt` at +16.1% over ZI (z=5.81)
