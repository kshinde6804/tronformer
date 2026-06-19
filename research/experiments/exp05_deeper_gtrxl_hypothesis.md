# exp05: Deeper GTrXL (num_layers=3)

## What is changing
Increase GTrXL depth from 2 to 3 layers (`--xfmr-layers 3`). All other hyperparameters
unchanged (d_model=128, nhead=4, dim_feedforward=256, max_seq=64, gru_bg=2.0).
No code changes required.

## Why
The original GTrXL paper (Parisotto et al. 2020) used 9 layers on Atari/DMLab. Our current
2-layer model is shallow relative to typical RL Transformers. With ~24 steps of context,
deeper layers may compose temporal patterns that 2 layers cannot: e.g., a 3rd layer can
reason about "arrivals where the midprice move (layer 1) coincided with volume imbalance
(layer 2)." With d_model=128, adding a 3rd layer adds ~600k parameters (+50% over 2-layer
model), which is modest on CPU.

Depth typically helps more than width for composing temporal abstractions. This is the
simplest high-expected-impact change available.

## Expected effect
+1 to +3% over ZI. Moderate-high confidence. Risk: deeper gating may produce instability
early in training; if loss diverges before 20k episodes, log as failed iteration.

## Previous best
Checkpoint: `out_exp_gtrxl_best.pt` at +16.1% over ZI (z=5.81)
