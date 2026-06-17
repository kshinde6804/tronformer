# GTrXL-style GRU Gating for TRONTransformerNet

## Core idea

Parisotto et al. (2020) "Stabilizing Transformers for Reinforcement Learning" showed that standard
Transformer residual connections (`x = x + sublayer(x)`) are unstable in RL optimization. The
Gated Transformer-XL (GTrXL) replaces these with GRU-type gating:

```
gate(x, y):  r = σ(Wr·y + Ur·x)
             z = σ(Wz·y + Uz·x − bg)   # bg init ≫ 0 → z near 0 → identity init
             h = tanh(Wg·y + Ug·(r⊙x))
             out = (1−z)·x + z·h
```

Applied at both the self-attention sublayer and the FFN sublayer. Combined with pre-norm
(already present in our setup as `norm_first=True`), this forms the full GTrXL architecture.

## Applicability to TRON

- **In scope**: purely a change to the recurrent block (replaces `nn.TransformerEncoder`
  with a custom gated stack; input projection and dueling heads untouched)
- **Identity initialization**: bg ≈ 2.0 initializes the gate near identity (z≈0.12),
  so training starts from a near-Markovian policy — same benefit that makes GTrXL
  competitive with LSTM from episode 1
- **Fit**: TRON is online RL on a market simulator with ~2000-step episodes. The
  instability that GTrXL addresses (gradient signal diluted through residual chains)
  is exactly the pathology expected here. Our LSTM already has gated memory; the
  Transformer variant lacks it, which likely explains the 5% gap vs. LSTM.
- **Caveat**: the TRON episode length T=2000 with max_seq=64 means the transformer
  sees at most 64 steps. Long-horizon memory issues are already truncated. GTrXL's
  advantage may be more about optimization stability than memory expressivity here.
  But stability is still the bottleneck (our xfmr trails the LSTM baseline).

## Expected impact

+2 to +6% over ZI (from 9.7% → 11.7–15.7%). GTrXL consistently beats standard
Transformer in RL settings; the effect is strongest when current Transformer lags
LSTM, which is our case.

## Implementation cost

Medium. Can't use `nn.TransformerEncoderLayer` (no hook for residual). Must write a
custom `GatedTransformerLayer` and `GatedTransformerEncoder` stack. ~50-80 LOC.
Self-contained in network.py. No changes to trainer, env, or eval.

## References

- Parisotto et al. (2020) "Stabilizing Transformers for Reinforcement Learning", ICML.
  https://arxiv.org/abs/1910.06764
- DI-engine reference implementation: https://opendilab.github.io/DI-engine/12_policies/gtrxl.html
- RLBenchNet (2025): GTrXL offers "smoother learning curves, lower variance, and slightly
  stronger final performance" vs Transformer-XL. https://arxiv.org/html/2505.15040v1
