# Training Recipe Improvements for TRONformer

## Ideas in scope (training recipe changes — Phase 1 allowed)

### 1. Context-encoder-specific learning rate (Luo et al. NeurIPS 2024)

"Efficient Recurrent Off-Policy RL Requires a Context-Encoder-Specific Learning Rate"
(https://github.com/FanmingL/Recurrent-Offpolicy-RL)

Key finding: the recurrent context encoder (Transformer or RNN) needs a *lower* learning rate
than the policy/value heads in off-policy recurrent RL. Standard uniform LR causes the encoder
to chase non-stationary targets from the replay buffer.

Proposal: use `optimizer = Adam([{'params': encoder_params, 'lr': lr * 0.1},
                                   {'params': head_params, 'lr': lr}])`

Applicability: our TRON uses a unified Q-net with input_proj + recurrent block + dueling heads.
The "encoder" is `input_proj + encoder (TransformerEncoder)`. The "heads" are `heads.adv_head`
and `heads.val_head`. Splitting LR is a training recipe change (Phase 1 in-scope) and doesn't
touch the architecture.

Expected impact: +1 to +4% over ZI. Risk: misidentifying which params are the encoder.
Implementation cost: low (one param group split in trainer.py + --encoder-lr-scale CLI flag).

### 2. Longer max_seq (context window)

Current: max_seq=64. Episode length: T=2000. TRON only sees last 64 arrivals.
Try max_seq=128 or 256 with no other changes. Larger context = more episodes per update
(replay buffer episodes are shorter-than-max_seq), so GPU/CPU mem and training time scale.

Expected impact: +1 to +3% if memory beyond 64 steps matters. On CPU this will be slow.
Risk: quadratic self-attention O(T²) doubles compute for 128→256.

Implementation cost: zero code change — just add `--xfmr-max-seq 128` to train command.

### 3. Attention entropy regularization (AMAGO 2024)

Add an auxiliary loss term `−λ * mean(entropy(attention_weights))` to encourage the
Transformer to distribute attention rather than collapse to attending only the last token.
This addresses a known failure mode of causal Transformers in RL (attending only to t−1).

Implementation: requires hooking into attention weights — more invasive with nn.TransformerEncoder.
Natural to add if we write a custom GTrXL layer (already exposes attention weights).

Expected impact: +0 to +2%. Modest. Better as a secondary modifier.

## Priority ranking (standalone)

1. Context-encoder-specific LR — pure recipe change, low risk, directly relevant finding
2. Longer max_seq — zero code, low risk, easy to test
3. Attention entropy reg — needs custom layer, save for after GTrXL experiment

## References

- Luo et al. NeurIPS 2024 https://arxiv.org/pdf/NeurIPS-2024-RESER (search: "context encoder specific learning rate")
- AMAGO (2024) ICLR https://proceedings.iclr.cc/paper_files/paper/2024/file/7204434dcb9383a1454dc1e97e58ea9c-Paper-Conference.pdf
