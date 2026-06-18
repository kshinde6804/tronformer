# Experiment Log

| Experiment | Change | Prev Best % | New % | z | Date | Decision |
|---|---|---|---|---|---|---|
| exp01_gtrxl_gating | GRU-gated Transformer (GTrXL), bg=2.0 | 9.7% | 16.1% | 5.81 | 2026-06-17 | keep |
| exp02_encoder_lr | encoder-specific LR (scale=0.1, GTrXL arch) | 16.1% | 15.7% | 5.65 | 2026-06-17 | discard |
| exp03_max_seq_128 | max_seq 64→128 (GTrXL arch) | 16.1% | 16.1% | 5.81 | 2026-06-18 | discard (null — context never truncates at 64) |
