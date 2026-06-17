# Baseline Metrics (Step 0 — written once, never modified)

Verified on 2026-06-16 with `eval_only.py --episodes 10000 --seed 42`
after Step 0 scaffold refactor (shared InputProjection + FactoredDuelingHeads).
Key remap (`remap_state_dict`) confirmed correct via `strict=True` load.

## LSTM final checkpoint (`tron_env_c_v2_final.pt`)

```
ZI   mean PnL = +1253.59  sd=2423.76  sem=24.24
TRON mean PnL = +1410.24  sd=2525.71  sem=25.26
delta         = +156.65  sem=35.01  z=4.47
             = +12.5% over ZI
```

Note: README originally reported +14.7% with ZI mean = +1216.94 (different RNG state
at time of authoring). Both are within eval noise (SEM(delta) ≈ 35 → ~2.8% per sigma).
Going forward, the eval_only.py with --seed 42 is authoritative; ZI = +1253.59 is the
stable baseline.

## Transformer best checkpoint (`tron_env_c_xfmr_best.pt`)

```
ZI   mean PnL = +1253.59  sd=2423.76  sem=24.24
TRON mean PnL = +1375.79  sd=2508.71  sem=25.09
delta         = +122.20  sem=34.88  z=3.50
             = +9.7% over ZI
```

This exactly matches the RESEARCH.md stated value. ✓

## GTrXL best checkpoint (`out_exp_gtrxl_best.pt`) — exp01, 2026-06-17

```
ZI   mean PnL = +1253.59  sd=2423.76  sem=24.24
TRON mean PnL = +1455.96  sd=2503.66  sem=25.04
delta         = +202.37  sem=34.85  z=5.81
             = +16.1% over ZI
```

Best checkpoint saved at ep65000. Beats LSTM ceiling (+12.5%) by +3.6 pp. ✓

## Reference for experiments

- **Beat this to improve:** +16.1% over ZI (z=5.81) from `out_exp_gtrxl_best.pt`
- **LSTM ceiling reference:** +12.5% over ZI (z=4.47) from `tron_env_c_v2_final.pt`
- **Step 0 Transformer baseline:** +9.7% over ZI (z=3.50) from `tron_env_c_xfmr_best.pt`
- ZI baseline: mean PnL = +1253.59 (used for all future % calculations with this seed)
- Eval config: `eval_only.py --episodes 10000 --seed 42`
