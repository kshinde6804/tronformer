# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
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

## Key Commands

**Eval (headline LSTM result):**
```bash
./.venv/bin/python -u -m marketsim.tron.eval_only \
  --checkpoint tron_env_c_v2_final.pt --episodes 10000 --seed 42
```

**ZI-only baseline:**
```bash
./.venv/bin/python -u -m marketsim.tron.zi_only --episodes 10000 --seed 42
```

**Train (LSTM):**
```bash
./.venv/bin/python -u -m marketsim.tron.train \
  --episodes 100000 --eps-decay 40000 --buffer-size 50000 \
  --lr 1e-4 --target-tau 0.005 --reward-clip 5.0 \
  --checkpoint out_final.pt --best-checkpoint out_best.pt
```

**Train (Transformer):** add `--arch transformer` (optionally `--xfmr-layers`, `--xfmr-heads`, `--xfmr-max-seq`).

`eval_only.py` reads `arch` from the checkpoint dict — no extra flag needed at eval time.

## Architecture

### PyMarketSim (pymarketsim/)
Event-driven limit order book simulator. Core abstractions:
- `marketsim/market/market.py` — `Market` manages the event queue, four-heap order book, and matching
- `marketsim/fourheap/fourheap.py` — price-time priority matching (four heaps: buy/sell matched/unmatched)
- `marketsim/fundamental/` — mean-reverting Gaussian fundamentals (`LazyGaussianMeanReverting` is the one used by TRON)
- `marketsim/agent/zero_intelligence_agent.py` — ZI agent (ZI Eq. 5 + η-threshold rule); used as both background population and the baseline

### TRON Environment (pymarketsim/marketsim/wrappers/tron_env.py)
`TRONEnv` is a Gymnasium environment registered as `PyMarketSim-TRON-v0`. It runs 24 ZI background agents + 1 RL-controlled TRON trader. Import `marketsim.wrappers` to trigger registration.

- **Action space:** `MultiDiscrete([21, 21])` — `(s_idx, eta_idx)` factored discrete. `s_grid` ∈ {0, 50, …, 1000}; `eta_grid` ∈ {0.0, 0.05, …, 1.0}.
- **Observation (10 floats):** time remaining fraction, fundamental estimate, best bid, best ask, position/q_max, PV for next buy, PV for next sell, midprice move, volume imbalance, side indicator.
- **Reward:** per-arrival ΔMtM (cash + position × fundamental estimate + private values), divided by `reward_scale`. Terminal step settles on the realized final fundamental.

### Networks (pymarketsim/marketsim/tron/network.py)
Two variants with identical `forward(obs, hidden) → (q_s, q_eta, new_hidden)`:
- **`TRONNet` (LSTM):** `Linear→ReLU→LSTM` encoder, dueling advantage+value heads for each action factor. `hidden = (h, c)` tuple.
- **`TRONTransformerNet`:** replaces LSTM with a causal pre-norm `TransformerEncoder` (sinusoidal PE, `is_causal=True`). `hidden` is the pre-positional embedding cache `(B, T_past, d_model)`.

`build_network(arch, input_dim, ...)` selects between them. Both use factored dueling heads: `Q_s = V_s + (A_s − mean(A_s))`, same for `Q_eta`.

### Trainer (pymarketsim/marketsim/tron/trainer.py)
R2D2-style single-process recurrent DQN. `ReplayBuffer` stores whole episodes (variable-length); `_pad_episodes` pads to `(B, T_max, …)` with a mask for loss weighting.

Five stabilization fixes vs. paper-naive R2D2:
1. Huber (smooth L1) loss — bounds gradient on outlier TD targets
2. Soft Polyak target updates (`tau=0.005`) — eliminates hard-sync discontinuities
3. Per-step reward clipping to ±5
4. Lower LR (`1e-4` instead of `5e-4`)
5. Best-checkpoint tracking by rolling mean PnL over a window

### Checkpoint format
`torch.save({"q_net": state_dict, "cfg": cfg.__dict__}, path)` — `cfg` includes `arch` and `arch_kwargs`, so `eval_only.py` can reconstruct the right network without any extra flags.

## Env C defaults (Mascioli et al., Table 3)
`lam=0.012, shock_var=2e4, pv_var=2e7, T=2000, N=25, q_max=10, f_bar=1e5, kappa=0.01`  
ZI background: `shade=[450, 540], eta=0.5`

See `RESEARCH.md` for the iterative TRONformer improvement workflow.
