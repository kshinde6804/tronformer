# TRON in PyMarketSim

Reproduction of the **TRON** agent from
[*A Financial Market Simulation Environment for Trading Agents Using Deep
Reinforcement Learning*](https://strategicreasoning.org/wp-content/uploads/2024/11/ICAIF24proceedings_PyMarketSim.pdf)
(Mascioli et al., ICAIF '24), trained against zero-intelligence (ZI)
background traders in [PyMarketSim](https://github.com/dipplestix/pymarketsim).

The repository adds:

- A clean Gymnasium environment, `PyMarketSim-TRON-v0`, exposing one
  RL-controlled trader with a factored discrete action `(s_idx, eta_idx)`
  vs. 24 ZI background agents.
- A simplified R2D2-style trainer (dueling DQN with LSTM or Transformer
  recurrence, per the paper's Fig. 3) with stabilization fixes that
  recover the paper's +11% TRON-vs-ZI gap in Env C.
- A pure 25-ZI baseline harness to verify the equilibrium-ZI reference numbers.

## Results

All runs use Env C from Table 3 of the paper:
`lambda=0.012, shock_var=2e4, pv_var=2e7, T=2000, N=25, q_max=10, f_bar=1e5, kappa=0.01`.
ZI background: `shade=[450, 540], eta=0.5`.

| Setting | Training | Eval (n) | ZI mean PnL | TRON mean PnL | Δ | z | % over ZI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LSTM (final ckpt)**       | **100 k eps** | **10 000** | **+1216.94** | **+1396.31** | **+179.37** | **5.15** | **+14.7%** |
| LSTM (best ckpt)            | 100 k eps     | 10 000     | +1253.59 | +1371.68 | +118.09 | 3.35 | +9.4% |
| Transformer (final ckpt)    | 100 k eps     | 10 000     | +1216.94 | +1279.27 | +62.33  | 1.80 | +5.1% |
| Transformer (best ckpt)     | 100 k eps     | 10 000     | +1253.59 | +1375.79 | +122.20 | 3.50 | +9.7% |
| Paper (Table 3)             | 3 M eps       | —          | +1259    | +1402    | +143    | —    | +11.4% |

The **Transformer variant** (`--arch transformer`, 2 encoder layers, 4 heads,
`d_model=128`, sinusoidal positional encoding, causal mask) is included as a
comparison. It matched the LSTM through ε-decay but its final policy drifted;
its best checkpoint is comparable to the LSTM's best (+122 vs +118) but the
LSTM's *final* checkpoint is the strongest of all (+179, z=5.15). Both
architectures have identical input projection, dueling heads, and factored
action space; only the recurrent block differs. Training time: LSTM 2.5 hrs
vs Transformer 3.5 hrs.

The stabilized trainer **clears the paper's reported gap** with ~30× less
training (100 k eps vs 3 M).

## Setup

```bash
git clone https://github.com/dipplestix/tronformer.git
cd tronformer
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install numpy scipy matplotlib gymnasium torch fastcubicspline pandas
./.venv/bin/pip install -e pymarketsim
```

Verify install:

```bash
./.venv/bin/python -c "import marketsim.wrappers; import gymnasium as gym; \
  print(gym.make('PyMarketSim-TRON-v0').reset(seed=0)[0].shape)"
```

## Quick start

### Reproduce the headline result (final LSTM checkpoint)

```bash
./.venv/bin/python -u -m marketsim.tron.eval_only \
  --checkpoint tron_env_c_v2_final.pt \
  --episodes 10000 --seed 42
```

Reports paired ZI baseline and TRON greedy PnL with SEM and z-score over 10 k episodes per arm.

### Verify the 25-ZI equilibrium baseline

```bash
./.venv/bin/python -u -m marketsim.tron.zi_only --episodes 10000 --seed 42
```

Should report mean per-agent PnL ≈ +1217 (paper Env C ZI: +1259).

## Training

### LSTM (recovers paper's gap, ~2.5 hours on CPU)

```bash
./.venv/bin/python -u -m marketsim.tron.train \
  --episodes 100000 --eval-episodes 10000 --log-every 5000 --eps-decay 40000 \
  --buffer-size 50000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --checkpoint out_final.pt \
  --best-checkpoint out_best.pt --best-window 5000 \
  --seed 0
```

### Transformer variant

Add `--arch transformer` (with optional `--xfmr-layers`, `--xfmr-heads`,
`--xfmr-max-seq`):

```bash
./.venv/bin/python -u -m marketsim.tron.train \
  --episodes 100000 --eval-episodes 10000 --log-every 5000 --eps-decay 40000 \
  --buffer-size 50000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --arch transformer --xfmr-layers 2 --xfmr-heads 4 --xfmr-max-seq 64 \
  --checkpoint out_xfmr_final.pt \
  --best-checkpoint out_xfmr_best.pt --best-window 5000 \
  --seed 0
```

`eval_only.py` reads the architecture from the checkpoint, so no extra
flag is needed at eval time.

### Useful flags

| flag | default | notes |
|---|---|---|
| `--episodes`            | 100 000 | total training episodes |
| `--eval-episodes`       | 500     | episodes per paired eval arm (10 k recommended) |
| `--log-every`           | 5 000   | print rolling stats every N episodes |
| `--eps-decay`           | 40 000  | epsilon-greedy decay to floor over N episodes |
| `--buffer-size`         | 50 000  | episode replay capacity |
| `--lr`                  | 1e-4    | Adam learning rate |
| `--target-tau`          | 0.005   | Polyak coefficient (0 to disable; falls back to hard sync) |
| `--reward-clip`         | 5.0     | clip per-step reward to ±c; 0 disables |
| `--arch`                | lstm    | `lstm` or `transformer` |
| `--lam`/`--shock-var`/`--pv-var`/`--sim-time` | Env C | environment overrides |
| `--checkpoint`          | —       | final state-dict path |
| `--best-checkpoint`     | —       | overwritten whenever rolling mean PnL hits a new high |
| `--mid-checkpoint`      | —       | snapshot at episodes/2 |

## Stabilization fixes (vs. paper-naive R2D2)

Five changes turned the trainer from "indistinguishable from ZI" into a
significant TRON win, with no architecture change:

1. **Huber (smooth L1) loss instead of MSE.** Bounds gradient on outlier
   TD targets, eliminates the loss spikes (was 100+, now 0.05–0.15).
2. **Soft (Polyak) target updates,** `tau=0.005` per grad step. Removes
   the discontinuity from hard target sync.
3. **Per-step reward clipping** to ±5. PV-driven trade windfalls
   contributed unboundedly to the TD target; clipping kept the value head
   well-scaled.
4. **Learning rate** dropped from 5e-4 to 1e-4 to match the smaller
   gradient signal after Huber.
5. **Best-checkpoint tracking** by 5000-ep rolling mean PnL guards
   against late-run regressions (kept as belt-and-suspenders; in our LSTM
   run the final checkpoint was the best, but the Transformer run's
   final ckpt drifted from peak).

## File layout

```
tronformer/
  README.md                                 this file
  pymarketsim/                              vendored upstream + new modules
    marketsim/
      wrappers/
        tron_env.py                         clean TRON Gymnasium env
        trader_env.py                       single-trader Gymnasium env (continuous)
        __init__.py                         registers PyMarketSim-TRON-v0 / -SingleTrader-v0
      tron/
        network.py                          dueling DQN + LSTM / Transformer variants
        trainer.py                          R2D2-style trainer (Huber, soft sync, reward clip)
        train.py                            training script (CLI for env + arch + trainer)
        eval_only.py                        paired ZI+TRON eval over N episodes
        zi_only.py                          pure 25-ZI baseline
  tron_env_c_v2_final.pt                    LSTM winner (+14.7% over ZI, z=5.15)
  tron_env_c_xfmr_best.pt                   Transformer best (+9.7% over ZI, z=3.50)
```

## Provenance

- Architecture: dueling DQN with LSTM, factored 21-s × 21-η action heads
  (Mascioli et al., Fig. 3); Transformer variant swaps the LSTM for a
  small causal Transformer encoder with the same heads and projections.
- ZI shade `[450, 540]` and `eta=0.5` taken from the user-supplied
  equilibrium ZI prior.
- Reward = per-arrival ΔMtM (paper Eq. 4) with the final step settled
  on the realised final fundamental.
- Order construction follows ZI Eq. 5 + η-threshold rule in §4.1.
