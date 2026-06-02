"""Load the trained TRON checkpoint and run a larger eval to tighten the SEM."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import gymnasium as gym

import marketsim.wrappers  # noqa: F401  registers PyMarketSim-TRON-v0
from marketsim.tron.train import evaluate, evaluate_zi_baseline, make_env
from marketsim.tron.trainer import TRONTrainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.012)
    parser.add_argument("--shock-var", type=float, default=2e4)
    parser.add_argument("--pv-var", type=float, default=2e7)
    parser.add_argument("--sim-time", type=int, default=2000)
    args = parser.parse_args()

    env = make_env(
        seed=args.seed,
        zi_eta=0.5,
        lam=args.lam,
        shock_var=args.shock_var,
        pv_var=args.pv_var,
        sim_time=args.sim_time,
    )
    # Peek at the checkpoint to pick the right architecture before constructing the trainer.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_in_ckpt = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    arch = cfg_in_ckpt.get("arch", "lstm")
    arch_kwargs = cfg_in_ckpt.get("arch_kwargs", None)
    trainer = TRONTrainer(
        env,
        cfg=TrainerConfig(arch=arch, arch_kwargs=arch_kwargs),
        seed=args.seed,
    )
    trainer.load(args.checkpoint)
    trainer.q_net.eval()

    np.random.seed(args.seed)

    print(f"--- ZI [450,540] baseline over {args.episodes} eval episodes ---", flush=True)
    t0 = time.time()
    zi = evaluate_zi_baseline(env, args.episodes)
    print(f"  {zi}    took {time.time()-t0:.1f}s", flush=True)

    print(f"--- TRON greedy over {args.episodes} eval episodes ---", flush=True)
    t0 = time.time()
    tron = evaluate(env, trainer, args.episodes)
    print(f"  {tron}    took {time.time()-t0:.1f}s", flush=True)

    n = args.episodes
    sem_zi = zi["std_pnl"] / np.sqrt(n)
    sem_tron = tron["std_pnl"] / np.sqrt(n)
    delta = tron["mean_pnl"] - zi["mean_pnl"]
    sem_delta = float(np.sqrt(sem_zi**2 + sem_tron**2))
    z = delta / sem_delta if sem_delta > 0 else float("inf")

    print("--- summary ---")
    print(f"  n = {n} eval episodes per arm")
    print(f"  ZI   mean PnL = {zi['mean_pnl']:+.2f}  sd={zi['std_pnl']:.2f}  sem={sem_zi:.2f}")
    print(f"  TRON mean PnL = {tron['mean_pnl']:+.2f}  sd={tron['std_pnl']:.2f}  sem={sem_tron:.2f}")
    print(f"  delta         = {delta:+.2f}  sem={sem_delta:.2f}  z={z:.2f}")
    pct = 100.0 * delta / max(1.0, abs(zi["mean_pnl"]))
    print(f"                = {pct:+.1f}% over ZI")


if __name__ == "__main__":
    main()
