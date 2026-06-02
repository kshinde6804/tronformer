"""Pure 25-ZI simulation with shade [450, 540] and eta=0.5.

Uses the same market mechanics as the TRON training env — 24 background ZI
agents plus a 25th agent occupying the "TRON slot" that also plays ZI
(random s in [450, 540], eta=0.5). Reports per-agent PnL distribution
across episodes plus aggregate welfare.

Env parameters default to Env C (lam=0.012, shock_var=2e4, pv_var=2e7);
override via CLI.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from marketsim.tron.train import make_env


def run(
    num_episodes: int,
    seed: int = 0,
    lam: float = 0.012,
    shock_var: float = 2e4,
    pv_var: float = 2e7,
    sim_time: int = 2000,
) -> dict:
    env = make_env(
        seed=seed,
        zi_eta=0.5,
        lam=lam,
        shock_var=shock_var,
        pv_var=pv_var,
        sim_time=sim_time,
    )
    env_u = env.unwrapped

    np.random.seed(seed)

    s_grid = env_u.s_grid
    eta_grid = env_u.eta_grid
    eta_idx = int(np.argmin(np.abs(eta_grid - 0.5)))

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

        fund_final = env_u.market.get_final_fundamental()
        for aid in range(24):
            agent = env_u.agents[aid]
            pnls[ep, aid] = agent.get_pos_value() + agent.position * fund_final + agent.cash
        pnls[ep, 24] = env_u.get_terminal_pnl()

        if env_u.position != starting_pos:
            slot_fills += 1
        welfare[ep] = pnls[ep].sum()
    elapsed = time.time() - t0

    return {
        "elapsed_s": elapsed,
        "n_episodes": num_episodes,
        "per_agent_mean": pnls.mean(axis=0),
        "per_agent_sem": pnls.std(axis=0, ddof=1) / np.sqrt(num_episodes),
        "mean_pnl_all25": float(pnls.mean()),
        "sem_pnl_all25": float(pnls.mean(axis=0).std(ddof=1) / np.sqrt(25)),
        "welfare_mean": float(welfare.mean()),
        "welfare_sem": float(welfare.std(ddof=1) / np.sqrt(num_episodes)),
        "tron_slot_mean": float(pnls[:, 24].mean()),
        "tron_slot_sem": float(pnls[:, 24].std(ddof=1) / np.sqrt(num_episodes)),
        "bg_mean": float(pnls[:, :24].mean()),
        "bg_sem": float(pnls[:, :24].mean(axis=0).std(ddof=1) / np.sqrt(24)),
        "slot_fill_rate": slot_fills / num_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.012)
    parser.add_argument("--shock-var", type=float, default=2e4)
    parser.add_argument("--pv-var", type=float, default=2e7)
    parser.add_argument("--sim-time", type=int, default=2000)
    args = parser.parse_args()

    print(
        f"--- 25-ZI run: shade=[450,540], eta=0.5, "
        f"lam={args.lam}, T={args.sim_time}, sigma_s={args.shock_var}, "
        f"sigma_pv={args.pv_var}, kappa=0.01, mean=1e5 ---",
        flush=True,
    )
    stats = run(
        args.episodes,
        seed=args.seed,
        lam=args.lam,
        shock_var=args.shock_var,
        pv_var=args.pv_var,
        sim_time=args.sim_time,
    )
    print(f"  elapsed: {stats['elapsed_s']:.1f}s for {stats['n_episodes']} episodes")
    print(f"  mean PnL per agent (averaged over 25 agents) = "
          f"{stats['mean_pnl_all25']:+.2f}  sem={stats['sem_pnl_all25']:.2f}")
    print(f"  bg ZI (24 agents) mean PnL    = {stats['bg_mean']:+.2f}  sem={stats['bg_sem']:.2f}")
    print(f"  TRON-slot ZI (1 agent) PnL    = {stats['tron_slot_mean']:+.2f}  sem={stats['tron_slot_sem']:.2f}")
    print(f"  per-episode welfare (sum across 25) mean = "
          f"{stats['welfare_mean']:+.2f}  sem={stats['welfare_sem']:.2f}")
    print(f"  slot fill rate (TRON-slot traded): {stats['slot_fill_rate']:.3f}")
    print("--- per-agent breakdown (mean ± sem) ---")
    for aid in range(25):
        print(f"  agent {aid:>2d}: {stats['per_agent_mean'][aid]:+10.2f}  ± {stats['per_agent_sem'][aid]:6.2f}")


if __name__ == "__main__":
    main()
