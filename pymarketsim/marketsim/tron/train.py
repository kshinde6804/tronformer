"""Train + evaluate a TRON agent against ZI background traders.

Defaults to Env C from Table 3 of Mascioli et al., ICAIF '24:
    N = 25 agents (24 ZI + 1 TRON), q_max = 10, f_bar = 1e5, kappa = 0.01,
    T = 2000, lambda = 0.012, shock_var = 2e4, pv_var = 2e7.
ZI background: shade = [450, 540], eta = 0.5.

Override any env parameter via CLI flags (--lam, --shock-var, --pv-var, --sim-time).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym

import marketsim.wrappers  # registers PyMarketSim-TRON-v0  # noqa: F401
from marketsim.tron.trainer import TRONTrainer, TrainerConfig


def make_env(
    seed: int | None = None,
    zi_eta: float = 0.5,
    lam: float = 0.012,
    shock_var: float = 2e4,
    pv_var: float = 2e7,
    sim_time: int = 2_000,
) -> gym.Env:
    env = gym.make(
        "PyMarketSim-TRON-v0",
        num_background_agents=24,
        sim_time=sim_time,
        lam=lam,
        mean=1e5,
        kappa=0.01,
        shock_var=shock_var,
        pv_var=pv_var,
        zi_shade=(450.0, 540.0),
        zi_eta=zi_eta,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate(env: gym.Env, trainer: TRONTrainer, num_episodes: int) -> dict:
    pnls = []
    returns = []
    positions = []
    fill_counts = 0
    decisions = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        hidden = trainer.q_net.initial_hidden(batch_size=1, device=trainer.device)
        done = False
        ep_r = 0.0
        starting_pos = env.unwrapped.position
        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).to(trainer.device).view(1, 1, -1)
                q_s, q_eta, hidden = trainer.q_net(obs_t, hidden)
                a_s = int(q_s[0, -1].argmax().item())
                a_eta = int(q_eta[0, -1].argmax().item())
            action = np.array([a_s, a_eta], dtype=np.int64)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
            decisions += 1
        end_pos = env.unwrapped.position
        if end_pos != starting_pos:
            fill_counts += 1
        pnls.append(env.unwrapped.get_terminal_pnl())
        returns.append(ep_r)
        positions.append(end_pos)
    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "mean_return": float(np.mean(returns)),
        "mean_position": float(np.mean(positions)),
        "fill_rate": fill_counts / max(1, num_episodes),
        "decisions": decisions,
    }


def evaluate_zi_baseline(env: gym.Env, num_episodes: int, zi_eta: float = 0.5) -> dict:
    """Drop in a ZI [450, 540] action by choosing s ~ U(450, 540) and a fixed eta."""
    pnls = []
    for _ in range(num_episodes):
        env.reset()
        env_unwrapped = env.unwrapped
        s_grid = env_unwrapped.s_grid
        eta_grid = env_unwrapped.eta_grid
        eta_idx = int(np.argmin(np.abs(eta_grid - zi_eta)))
        done = False
        while not done:
            target_s = np.random.uniform(450.0, 540.0)
            s_idx = int(np.argmin(np.abs(s_grid - target_s)))
            action = np.array([s_idx, eta_idx], dtype=np.int64)
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc
        pnls.append(env_unwrapped.get_terminal_pnl())
    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="tron_final.pt")
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--eps-decay", type=int, default=40_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--mid-checkpoint", type=str, default=None,
                        help="Save an intermediate checkpoint at episodes/2.")
    parser.add_argument("--lam", type=float, default=0.012)
    parser.add_argument("--shock-var", type=float, default=2e4)
    parser.add_argument("--pv-var", type=float, default=2e7)
    parser.add_argument("--sim-time", type=int, default=2000)
    parser.add_argument("--best-checkpoint", type=str, default=None)
    parser.add_argument("--best-window", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--arch", type=str, default="lstm",
                        choices=["lstm", "transformer", "gated-transformer"])
    parser.add_argument("--xfmr-layers", type=int, default=2)
    parser.add_argument("--xfmr-heads", type=int, default=4)
    parser.add_argument("--xfmr-max-seq", type=int, default=64)
    parser.add_argument("--gtrxl-bg", type=float, default=2.0,
                        help="GRU gate bias init for gated-transformer (higher → closer to identity at init)")
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0,
                        help="LR multiplier for encoder params (input_proj+recurrent) vs heads. "
                             "Luo et al. NeurIPS 2024 recommends 0.1.")
    args = parser.parse_args()

    env_kwargs = dict(
        zi_eta=0.5,
        lam=args.lam,
        shock_var=args.shock_var,
        pv_var=args.pv_var,
        sim_time=args.sim_time,
    )
    train_env = make_env(seed=args.seed, **env_kwargs)
    eval_env = make_env(seed=args.seed + 10_000, **env_kwargs)

    arch_kwargs = None
    if args.arch == "transformer":
        arch_kwargs = dict(
            num_layers=args.xfmr_layers,
            nhead=args.xfmr_heads,
            max_seq=args.xfmr_max_seq,
        )
    elif args.arch == "gated-transformer":
        arch_kwargs = dict(
            num_layers=args.xfmr_layers,
            nhead=args.xfmr_heads,
            max_seq=args.xfmr_max_seq,
            gru_bg=args.gtrxl_bg,
        )
    cfg = TrainerConfig(
        device=args.device,
        eps_decay_episodes=args.eps_decay,
        buffer_size=args.buffer_size,
        lr=args.lr,
        encoder_lr_scale=args.encoder_lr_scale,
        target_tau=args.target_tau,
        reward_clip=args.reward_clip,
        arch=args.arch,
        arch_kwargs=arch_kwargs,
    )
    trainer = TRONTrainer(train_env, cfg=cfg, seed=args.seed)

    print(f"--- baseline ZI [450,540] eval over {args.eval_episodes} episodes ---")
    np.random.seed(args.seed + 999)
    zi_stats = evaluate_zi_baseline(eval_env, args.eval_episodes)
    print(zi_stats)

    print(f"--- training TRON for {args.episodes} episodes ---")
    t0 = time.time()
    train_kwargs = dict(log_every=args.log_every)
    if args.best_checkpoint:
        Path(args.best_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        train_kwargs["best_checkpoint"] = args.best_checkpoint
        train_kwargs["best_window"] = args.best_window
    if args.mid_checkpoint:
        half = args.episodes // 2
        trainer.train(half, **train_kwargs)
        Path(args.mid_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.mid_checkpoint)
        print(f"saved mid checkpoint to {args.mid_checkpoint} at ep {half}")
        trainer.train(args.episodes - half, **train_kwargs)
    else:
        trainer.train(args.episodes, **train_kwargs)
    print(f"training took {time.time() - t0:.1f}s")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")

    print(f"--- TRON eval (greedy) over {args.eval_episodes} episodes ---")
    tron_stats = evaluate(eval_env, trainer, args.eval_episodes)
    print(tron_stats)

    print("--- summary ---")
    print(f"  ZI baseline mean PnL: {zi_stats['mean_pnl']:+.2f}  (sd {zi_stats['std_pnl']:.2f})")
    print(f"  TRON greedy mean PnL: {tron_stats['mean_pnl']:+.2f}  (sd {tron_stats['std_pnl']:.2f})")
    delta = tron_stats["mean_pnl"] - zi_stats["mean_pnl"]
    pct = 100.0 * delta / max(1.0, abs(zi_stats["mean_pnl"]))
    print(f"  delta = {delta:+.2f} ({pct:+.1f}% over ZI)")


if __name__ == "__main__":
    main()
