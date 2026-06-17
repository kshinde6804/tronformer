# Rotary Positional Encoding (RoPE) for TRONTransformerNet

## Core idea

Su et al. (2024) "RoFormer: Enhanced Transformer with Rotary Position Embedding". Instead of
adding sinusoidal positional vectors to token embeddings, RoPE encodes position by rotating
query/key vectors in the attention computation:

```
RoPE(q, pos) = q ⊗ cos(pos·θ) + rotate_half(q) ⊗ sin(pos·θ)
```

Applied to Q and K (not V) in each attention head. Encodes *relative* position implicitly:
`<RoPE(q_i), RoPE(k_j)>` depends only on `q_i`, `k_j`, and `i−j` (the lag).

## Applicability to TRON

- **In scope**: only modifies how positional information is injected inside the attention
  sublayer — purely a recurrent block change
- **Fit**: market microstructure features like "midprice move" and "volume imbalance" are
  relative-order-sensitive (recency matters). RoPE's relative-position bias may help the
  Transformer attend to recent arrivals more naturally than sinusoidal absolute PE.
- **Caveat**: our max_seq=64 is short. Sinusoidal PE works fine for short sequences; the
  relative-position advantage of RoPE is most pronounced at longer context lengths. Impact
  may be modest here. Also, implementing RoPE from scratch requires modifying attention
  internals — can't easily plug into `nn.MultiheadAttention` without a custom layer.
  If combined with the GTrXL custom layer (which already requires a custom attention),
  adding RoPE is nearly free. As a standalone change it costs more.

## Expected impact

+0 to +2% over ZI. Moderate confidence. Better as a secondary change combined with GTrXL.

## Implementation cost

High standalone (needs custom attention), Low if combined with GTrXL gating layer.

## References

- Su et al. (2024) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  https://arxiv.org/abs/2104.09864
- Used in LLaMA, GPT-NeoX, and virtually all modern LLMs for its simplicity and efficacy.
