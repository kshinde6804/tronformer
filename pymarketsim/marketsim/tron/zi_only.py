"""Pure 25-ZI simulation in Env A with shade [450, 540] and eta=0.5.

Uses the same market mechanics as the TRON training env — 24 background ZI
agents plus a 25th agent occupying the "TRON slot" that also plays ZI
(random s in [450, 540], eta=0.5). Reports per-agent PnL distribution
across episodes plus aggregate welfare.
"""

from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from marketsim.tron.train_env_a import make_env


def run(num_episodes: int, seed: int = 0) -> dict:
    env = make_env(seed=seed, zi_eta=0.5)
    env_u = env.unwrapped

    np.random.seed(seed)

    s_grid = env_u.s_grid
    eta_grid = env_u.eta_grid
    eta_idx = int(np.argmin(np.abs(eta_grid - 0.5)))

    # PnLs: 25 columns (24 bg ZI + 1 TRON-slot ZI) per episode.
    pnls = np.zeros((num_episodes, 25), dtype=np.float64)
    welfare = np.zeros(num_episodes, dtype=np.float64)
    slot_fills = 0

    t0 = time.time()
    for ep in range(num_episodes):
        env.reset()
        starting_pos = env_u.position
        done = False
        while not done:
            target_s = np.random.uniform(450.0, 540.0)
            s_idx = int(np.argmin(np.abs(s_grid - target_s)))
            action = np.array([s_idx, eta_idx], dtype=np.int64)
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc

        # Final fundamental for portfolio settlement.
        fund_final = env_u.market.get_final_fundamental()
        # Background ZI PnLs
        for aid in range(24):
            agent = env_u.agents[aid]
            pnls[ep, aid] = agent.get_pos_value() + agent.position * fund_final + agent.cash
        # TRON-slot (acting as ZI) PnL
        pnls[ep, 24] = env_u.get_terminal_pnl()

        if env_u.position != starting_pos:
            slot_fills += 1
        welfare[ep] = pnls[ep].sum()
    elapsed = time.time() - t0

    return {
        "elapsed_s": elapsed,
        "n_episodes": num_episodes,
        "per_agent_mean": pnls.mean(axis=0),                # (25,)
        "per_agent_sem": pnls.std(axis=0, ddof=1) / np.sqrt(num_episodes),
        "mean_pnl_all25": float(pnls.mean()),
        "sem_pnl_all25": float(pnls.mean(axis=0).std(ddof=1) / np.sqrt(25)),  # sem across agents
        "mean_per_agent_pnl_per_ep_sem": float(pnls.flatten().std(ddof=1) / np.sqrt(num_episodes * 25)),
        "welfare_mean": float(welfare.mean()),
        "welfare_sem": float(welfare.std(ddof=1) / np.sqrt(num_episodes)),
        "tron_slot_mean": float(pnls[:, 24].mean()),
        "tron_slot_sem": float(pnls[:, 24].std(ddof=1) / np.sqrt(num_episodes)),
        "bg_mean": float(pnls[:, :24].mean()),
        "bg_sem": float(pnls[:, :24].mean(axis=0).std(ddof=1) / np.sqrt(24)),
        "slot_fills": slot_fills,
        "slot_fill_rate": slot_fills / num_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        f"--- 25-ZI run: shade=[450,540], eta=0.5, Env A "
        f"(lam=5e-4, T=2000, sigma_s=1e6, sigma_pv=5e6, kappa=0.01, mean=1e5) ---",
        flush=True,
    )
    stats = run(args.episodes, seed=args.seed)
    print(f"  elapsed: {stats['elapsed_s']:.1f}s for {stats['n_episodes']} episodes")
    print(f"  mean PnL per agent (averaged over 25 agents) = "
          f"{stats['mean_pnl_all25']:+.2f}  sem(across-agent-mean)={stats['sem_pnl_all25']:.2f}")
    print(f"  per-episode-per-agent PnL sem = {stats['mean_per_agent_pnl_per_ep_sem']:.2f}")
    print(f"  bg ZI (24 agents) mean PnL    = {stats['bg_mean']:+.2f}  sem={stats['bg_sem']:.2f}")
    print(f"  TRON-slot ZI (1 agent) PnL   = {stats['tron_slot_mean']:+.2f}  sem={stats['tron_slot_sem']:.2f}")
    print(f"  per-episode welfare (sum across 25) mean = "
          f"{stats['welfare_mean']:+.2f}  sem={stats['welfare_sem']:.2f}")
    print(f"  slot fill rate (TRON-slot traded): {stats['slot_fill_rate']:.3f}")
    print("--- per-agent breakdown (mean ± sem) ---")
    for aid in range(25):
        print(f"  agent {aid:>2d}: {stats['per_agent_mean'][aid]:+10.2f}  ± {stats['per_agent_sem'][aid]:6.2f}")


if __name__ == "__main__":
    main()
