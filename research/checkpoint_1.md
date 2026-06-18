# Checkpoint 1 — 2026-06-18

## Results table
| Experiment | Change | Before % | After % | z | Decision |
|---|---|---|---|---|---|
| exp01_gtrxl_gating | GRU-gated Transformer (GTrXL), bg=2.0 | 9.7% | 16.1% | 5.81 | keep |
| exp02_encoder_lr | encoder-specific LR (scale=0.1, GTrXL arch) | 16.1% | 15.7% | 5.65 | discard |
| exp03_max_seq_128 | max_seq 64→128 (GTrXL arch) | 16.1% | 16.1% | 5.81 | discard |

## Current best
Checkpoint file: out_exp_gtrxl_best.pt
% over ZI: 16.1% (z=5.81)

## What was tried

**Exp01 — GTrXL GRU gating:** Replaced the standard residual connections in the Transformer encoder with GRU-type gates (Parisotto et al. 2020, bg=2.0 identity init). Pushed from +9.7% to +16.1%, exceeding the LSTM ceiling (+12.5%). The gating mechanism provides adaptive memory management that pure residual addition lacks — precisely the mechanism that makes LSTMs competitive with Transformers in RL settings.

**Exp02 — context-encoder-specific LR:** Split Adam optimizer with 10× lower LR for the encoder (input_proj + GatedTransformer) vs dueling heads, motivated by Luo et al. NeurIPS 2024. Scored +15.7%, within noise of +16.1% (difference = 0.4 pp, 1σ ≈ 2.8%). GTrXL gating may already provide the optimization stability that the lower encoder LR was intended to add, making the recipe change redundant on this architecture.

**Exp03 — max_seq=128:** Doubled the Transformer context window from 64 to 128 steps. Scored an identical +16.1% with exactly the same weights as exp01. Root cause: TRON receives only ~24 arrivals/episode in Env C (lam=0.012, T=2000). The context is never truncated at max_seq=64, so increasing to 128 makes no architectural difference. This is a true null result that rules out context length as a bottleneck.

## Phase status
Phase 1 — 2 consecutive non-improving iterations (Phase 2 trigger at 3)

## What's next

1. **GTrXL + RoPE positional encoding:** Replace sinusoidal absolute PE with rotary PE applied to attention Q/K vectors. Now that we have a custom attention layer (GatedTransformerLayer), adding RoPE is low-cost. The ~24-step context means relative-position bias may help the model weight recent arrivals more naturally than absolute positions. Notes: `research/notes/rope_positional_encoding.md`.

2. **Deeper GTrXL (num_layers=3 or 4):** Current model uses only 2 GTrXL layers. The original GTrXL paper used 9 layers. With only 24 steps of context, depth (not width) may be the capacity bottleneck. Expected: modest gain at low cost (one hyperparameter change). Risk: potential instability from deeper gating.

3. **Reduced max_seq (~24–32):** Exp03 confirmed the context is never truncated at 64. Reducing max_seq to 32 (matching the actual episode context length) reduces the positional encoding size and attention complexity, potentially acting as a regularizer. Low expected standalone impact but worth testing.

## Resume prompt

---
Continue the TRONformer research workflow defined in RESEARCH.md.
Current best Transformer checkpoint: out_exp_gtrxl_best.pt at 16.1% over ZI (z=5.81).
Phase: 1. Consecutive non-improving iterations: 2 (Phase 2 trigger at 3).
Start from the "What's next" section of research/checkpoint_1.md.
Proceed autonomously for 3 training cycles, then write checkpoint_2.md and push to master.
---
