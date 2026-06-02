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
- A simplified R2D2-style trainer (dueling DQN + LSTM per the paper's
  Fig. 3) with stabilization fixes that recover the paper's +11% TRON-vs-ZI
  gap in Env C.
- A pure 25-ZI baseline harness to verify the equilibrium-ZI reference numbers.

## Results

Env A: `lambda=5e-4, shock_var=1e6, pv_var=5e6, T=2000`.
Env C: `lambda=0.012, shock_var=2e4, pv_var=2e7, T=2000`.
Both: `N=25, q_max=10, f_bar=1e5, kappa=0.01`, ZI background `shade=[450,540], eta=0.5`.

| Setting | Training | Eval (n) | ZI mean PnL | TRON mean PnL | Δ | z | % over ZI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Env A, 25 pure ZI                  | —        | 10 000 | +107.88 | — | — | — | — |
| Env A, original trainer (MSE)      | 2 M eps  | 10 000 | +141.72 | +123.43 | −18.3 | −0.31 | indistinguishable |
| Env C, original trainer (MSE)      | 200 k eps| 10 000 | +1216.94 | +1202.79 | −14.2 | −0.42 | indistinguishable |
| **Env C, LSTM (final ckpt)**       | **100 k eps** | **10 000** | **+1216.94** | **+1396.31** | **+179.37** | **5.15** | **+14.7%** |
| Env C, LSTM (best ckpt)            | 100 k eps| 10 000 | +1253.59 | +1371.68 | +118.09 | 3.35 | +9.4% |
| Env C, Transformer (final ckpt)    | 100 k eps| 10 000 | +1216.94 | +1279.27 | +62.33  | 1.80 | +5.1% |
| Env C, Transformer (best ckpt)     | 100 k eps| 10 000 | +1253.59 | +1375.79 | +122.20 | 3.50 | +9.7% |
| Env C, paper (Table 3)             | 3 M eps  | —      | +1259    | +1402    | +143    | —    | +11.4% |

The **Transformer variant** (`--arch transformer`, 2 encoder layers, 4 heads, `d_model=128`,
sinusoidal positional encoding, causal mask) is included as a comparison. It
matched the LSTM through ε-decay but its final policy drifted; its best
checkpoint is comparable to the LSTM's best (+122 vs +118) but the LSTM's
*final* checkpoint is the strongest of all (+179, z=5.15). Both architectures
have identical input projection, dueling heads, and factored action space; only
the recurrent block differs. Training time: LSTM 2.5 hrs vs Transformer 3.5 hrs.

- 25 pure-ZI baseline (+107.88) matches paper Env A ZI (+106), confirming
  `shade=[450,540], eta=0.5` is the Env A equilibrium ZI.
- Stabilized trainer **clears the paper's reported gap** in Env C with
  ~30x less training (100 k eps vs 3 M).
- Env A is decision-sparse (~1 TRON arrival per episode); we did not
  reproduce the paper's small Env A gap there with available compute.

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

### Reproduce the headline Env C result (final checkpoint)

```bash
./.venv/bin/python -u -m marketsim.tron.eval_only \
  --checkpoint tron_env_c_v2_final.pt \
  --episodes 10000 --seed 42 \
  --lam 0.012 --shock-var 2e4 --pv-var 2e7 --sim-time 2000
```

Reports paired ZI baseline and TRON greedy PnL with SEM and z-score over 10 k episodes per arm.

### Verify the 25-ZI equilibrium baseline

```bash
./.venv/bin/python -u -m marketsim.tron.zi_only --episodes 10000 --seed 42
```

Should report mean per-agent PnL ≈ +108 in Env A (paper: +106).

## Training

### Env C (stabilized, ~2.5 hours, recovers paper's gap)

```bash
./.venv/bin/python -u -m marketsim.tron.train_env_a \
  --episodes 100000 --eval-episodes 10000 --log-every 5000 --eps-decay 40000 \
  --buffer-size 50000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --lam 0.012 --shock-var 2e4 --pv-var 2e7 --sim-time 2000 \
  --checkpoint out_env_c_final.pt \
  --best-checkpoint out_env_c_best.pt --best-window 5000 \
  --seed 0
```

### Env A (default; decision-sparse, harder to learn)

```bash
./.venv/bin/python -u -m marketsim.tron.train_env_a \
  --episodes 50000 --eval-episodes 10000 --log-every 2000 --eps-decay 20000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --checkpoint out_env_a.pt --seed 0
```

### Transformer variant

Add `--arch transformer` (with optional `--xfmr-layers`, `--xfmr-heads`,
`--xfmr-max-seq`):

```bash
./.venv/bin/python -u -m marketsim.tron.train_env_a \
  --episodes 100000 --eval-episodes 10000 --log-every 5000 --eps-decay 40000 \
  --buffer-size 50000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --arch transformer --xfmr-layers 2 --xfmr-heads 4 --xfmr-max-seq 64 \
  --lam 0.012 --shock-var 2e4 --pv-var 2e7 --sim-time 2000 \
  --checkpoint out_env_c_xfmr_final.pt \
  --best-checkpoint out_env_c_xfmr_best.pt --best-window 5000 \
  --seed 0
```

`eval_only.py` reads the architecture from the checkpoint, so no extra
flag is needed at eval time.

### Useful flags

| flag | default | notes |
|---|---|---|
| `--episodes`            | 50 000 | total training episodes |
| `--eval-episodes`       | 500    | episodes per paired eval arm (10 k recommended) |
| `--log-every`           | 500    | print rolling stats every N episodes |
| `--eps-decay`           | 20 000 | epsilon-greedy decay to floor over N episodes |
| `--buffer-size`         | 50 000 | episode replay capacity |
| `--lr`                  | 1e-4   | Adam learning rate |
| `--target-tau`          | 0.005  | Polyak coefficient (0 to disable; falls back to hard sync) |
| `--reward-clip`         | 5.0    | clip per-step reward to ±c; 0 disables |
| `--lam`/`--shock-var`/`--pv-var`/`--sim-time` | Env A | environment overrides |
| `--checkpoint`          | —      | final state-dict path |
| `--best-checkpoint`     | —      | overwritten whenever rolling mean PnL hits a new high |
| `--mid-checkpoint`      | —      | snapshot at episodes/2 |

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
   against late-run regressions (kept as belt-and-suspenders; in our run
   the final checkpoint was actually best).

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
        network.py                          dueling DQN + LSTM and Transformer variants
        trainer.py                          R2D2-style trainer (Huber, soft sync, reward clip)
        train_env_a.py                      training script (parameterised for any env)
        eval_only.py                        paired ZI+TRON eval over N episodes
        zi_only.py                          pure 25-ZI baseline
  tron_env_c_v2_final.pt                    LSTM winner (Env C, +14.7% over ZI, z=5.15)
  tron_env_c_xfmr_best.pt                   Transformer best (Env C, +9.7% over ZI, z=3.50)
```

## Provenance

- Architecture: dueling DQN with LSTM, factored 21-s × 21-η action heads
  (Mascioli et al., Fig. 3).
- ZI shade `[450, 540]` and `eta=0.5` taken from the user-provided
  equilibrium ZI prior; verified to match the paper's Env A ZI mean
  profit (+106) at +107.88.
- Reward = per-arrival ΔMtM (paper Eq. 4) with the final step settled
  on the realised final fundamental.
- Order construction follows ZI Eq. 5 + η-threshold rule in §4.1.
