"""R2D2-flavoured trainer for the TRON agent.

A simplified, single-process recurrent DQN: replay buffer over whole
episodes, factored Q-learning per action head, periodic target sync,
epsilon-greedy exploration per head.

The paper's full R2D2 uses distributed actors + prioritized replay; here we
keep the architecture and learning rule but drop the distributed plumbing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from marketsim.tron.network import build_network, remap_state_dict


@dataclass
class Transition:
    obs: np.ndarray         # (input_dim,)
    action: np.ndarray      # (2,) int  -- (s_idx, eta_idx)
    reward: float
    next_obs: np.ndarray    # (input_dim,)
    done: bool


@dataclass
class TrainerConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    encoder_lr_scale: float = 1.0       # multiplier for encoder (input_proj+recurrent) LR vs heads LR
    batch_size: int = 32                # number of episodes per update
    buffer_size: int = 50_000           # episodes
    target_tau: float = 0.005           # soft (Polyak) target update per grad step; set 0 to disable
    target_sync_every: int = 0          # if >0, hard target sync (disables soft updates)
    train_every: int = 4                # one update per N collected transitions
    min_buffer: int = 200               # episodes before training starts
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 20_000
    grad_clip: float = 10.0
    reward_clip: float = 5.0            # clip per-step reward into [-c, c]; set 0 to disable
    device: str = "cpu"
    huber: bool = True                  # smooth_l1 loss instead of MSE
    arch: str = "lstm"                  # "lstm" or "transformer"
    arch_kwargs: Optional[dict] = None  # extra kwargs forwarded to build_network


class ReplayBuffer:
    """Stores full episodes (variable-length lists of Transitions)."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[List[Transition]] = deque(maxlen=capacity)

    def add(self, episode: List[Transition]) -> None:
        if episode:
            self.buffer.append(episode)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[List[Transition]]:
        idx = rng.integers(0, len(self.buffer), size=batch_size)
        return [self.buffer[int(i)] for i in idx]


def _pad_episodes(
    episodes: List[List[Transition]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded tensors (B, T, ...) plus a mask (B, T)."""
    batch = len(episodes)
    max_len = max(len(ep) for ep in episodes)
    input_dim = episodes[0][0].obs.shape[0]

    obs = np.zeros((batch, max_len, input_dim), dtype=np.float32)
    next_obs = np.zeros((batch, max_len, input_dim), dtype=np.float32)
    action = np.zeros((batch, max_len, 2), dtype=np.int64)
    reward = np.zeros((batch, max_len), dtype=np.float32)
    done = np.zeros((batch, max_len), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=np.float32)

    for b, ep in enumerate(episodes):
        for t, tr in enumerate(ep):
            obs[b, t] = tr.obs
            next_obs[b, t] = tr.next_obs
            action[b, t] = tr.action
            reward[b, t] = tr.reward
            done[b, t] = float(tr.done)
            mask[b, t] = 1.0

    return (
        torch.from_numpy(obs).to(device),
        torch.from_numpy(action).to(device),
        torch.from_numpy(reward).to(device),
        torch.from_numpy(next_obs).to(device),
        torch.from_numpy(done).to(device),
        torch.from_numpy(mask).to(device),
    )


class TRONTrainer:
    def __init__(self, env, cfg: Optional[TrainerConfig] = None, seed: int = 0) -> None:
        self.env = env
        self.cfg = cfg or TrainerConfig()
        self.device = torch.device(self.cfg.device)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        input_dim = env.observation_space.shape[0]
        n_s, n_eta = env.action_space.nvec  # type: ignore[attr-defined]
        self.n_s = int(n_s)
        self.n_eta = int(n_eta)

        arch_kwargs = self.cfg.arch_kwargs or {}
        self.q_net = build_network(self.cfg.arch, input_dim=input_dim, n_s=self.n_s, n_eta=self.n_eta, **arch_kwargs).to(self.device)
        self.target_net = build_network(self.cfg.arch, input_dim=input_dim, n_s=self.n_s, n_eta=self.n_eta, **arch_kwargs).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        if self.cfg.encoder_lr_scale != 1.0:
            head_ids = {id(p) for p in self.q_net.heads.parameters()}
            enc_params = [p for p in self.q_net.parameters() if id(p) not in head_ids]
            head_params = list(self.q_net.heads.parameters())
            self.optimizer = optim.Adam([
                {"params": enc_params,  "lr": self.cfg.lr * self.cfg.encoder_lr_scale},
                {"params": head_params, "lr": self.cfg.lr},
            ])
        else:
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)

        self.episodes_seen = 0
        self.transitions_seen = 0
        self.grad_steps = 0
        self.last_train_at = 0

    # ---------- acting ----------

    def _epsilon(self) -> float:
        frac = min(1.0, self.episodes_seen / max(1, self.cfg.eps_decay_episodes))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def _select_action(
        self,
        obs: np.ndarray,
        hidden,
        epsilon: float,
    ):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device).view(1, 1, -1)
            q_s, q_eta, hidden = self.q_net(obs_t, hidden)
            # q_s, q_eta shaped (1, T, n_*); take the latest timestep's Q-values.
            q_s_now = q_s[0, -1]
            q_eta_now = q_eta[0, -1]
            if self.rng.random() < epsilon:
                a_s = int(self.rng.integers(0, self.n_s))
            else:
                a_s = int(q_s_now.argmax().item())
            if self.rng.random() < epsilon:
                a_eta = int(self.rng.integers(0, self.n_eta))
            else:
                a_eta = int(q_eta_now.argmax().item())
        return np.array([a_s, a_eta], dtype=np.int64), hidden

    # ---------- training ----------

    def _train_step(self) -> Optional[float]:
        if len(self.buffer) < self.cfg.min_buffer:
            return None

        episodes = self.buffer.sample(self.cfg.batch_size, self.rng)
        obs, action, reward, next_obs, done, mask = _pad_episodes(episodes, self.device)
        batch, max_len, _ = obs.shape

        if self.cfg.reward_clip > 0:
            reward = reward.clamp(-self.cfg.reward_clip, self.cfg.reward_clip)

        # Forward online net
        q_s_all, q_eta_all, _ = self.q_net(obs)
        a_s = action[..., 0:1]   # (B, T, 1)
        a_eta = action[..., 1:2]
        q_s_taken = q_s_all.gather(-1, a_s).squeeze(-1)        # (B, T)
        q_eta_taken = q_eta_all.gather(-1, a_eta).squeeze(-1)  # (B, T)

        with torch.no_grad():
            q_s_next, q_eta_next, _ = self.target_net(next_obs)
            v_s_next = q_s_next.max(dim=-1).values
            v_eta_next = q_eta_next.max(dim=-1).values
            target_s = reward + self.cfg.gamma * (1.0 - done) * v_s_next
            target_eta = reward + self.cfg.gamma * (1.0 - done) * v_eta_next

        denom = mask.sum().clamp(min=1.0)
        if self.cfg.huber:
            loss_s = (F.smooth_l1_loss(q_s_taken, target_s, reduction="none") * mask).sum() / denom
            loss_eta = (F.smooth_l1_loss(q_eta_taken, target_eta, reduction="none") * mask).sum() / denom
        else:
            diff_s = (q_s_taken - target_s) * mask
            diff_eta = (q_eta_taken - target_eta) * mask
            loss_s = diff_s.pow(2).sum() / denom
            loss_eta = diff_eta.pow(2).sum() / denom
        loss = loss_s + loss_eta

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.cfg.target_sync_every > 0:
            if self.grad_steps % self.cfg.target_sync_every == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        elif self.cfg.target_tau > 0:
            with torch.no_grad():
                tau = self.cfg.target_tau
                for p, p_t in zip(self.q_net.parameters(), self.target_net.parameters()):
                    p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        return float(loss.item())

    # ---------- run loop ----------

    def train(
        self,
        num_episodes: int,
        log_every: int = 500,
        best_checkpoint: Optional[str] = None,
        best_window: int = 5000,
    ) -> List[Tuple[int, float, float, Optional[float]]]:
        """Run training; return list of (episode_idx, return, terminal_pnl, last_loss).

        If `best_checkpoint` is provided, save a copy of the q_net every time
        the rolling-mean terminal PnL over `best_window` episodes hits a new
        all-time high — guards against late-training divergence.
        """
        history: List[Tuple[int, float, float, Optional[float]]] = []
        last_loss: Optional[float] = None
        return_window: Deque[float] = deque(maxlen=log_every)
        pnl_window: Deque[float] = deque(maxlen=log_every)
        best_pnl_window: Deque[float] = deque(maxlen=best_window)
        best_pnl: float = float("-inf")
        best_ep: int = -1

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            hidden = self.q_net.initial_hidden(batch_size=1, device=self.device)
            episode: List[Transition] = []
            done = False
            ep_return = 0.0
            eps = self._epsilon()

            while not done:
                action, hidden = self._select_action(obs, hidden, eps)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                episode.append(Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
                obs = next_obs
                ep_return += reward
                self.transitions_seen += 1
                if (
                    self.transitions_seen - self.last_train_at >= self.cfg.train_every
                    and len(self.buffer) >= self.cfg.min_buffer
                ):
                    last_loss = self._train_step()
                    self.last_train_at = self.transitions_seen

            self.buffer.add(episode)
            self.episodes_seen += 1
            terminal_pnl = float(self.env.unwrapped.get_terminal_pnl())  # type: ignore[attr-defined]
            return_window.append(ep_return)
            pnl_window.append(terminal_pnl)
            best_pnl_window.append(terminal_pnl)

            # Update best checkpoint once the window is full enough to be reliable.
            if (
                best_checkpoint is not None
                and len(best_pnl_window) >= best_window
                and (ep + 1) % max(1, log_every // 5) == 0
            ):
                rolling = float(np.mean(best_pnl_window))
                if rolling > best_pnl:
                    best_pnl = rolling
                    best_ep = ep + 1
                    self.save(best_checkpoint)

            if (ep + 1) % log_every == 0:
                avg_r = float(np.mean(return_window))
                avg_pnl = float(np.mean(pnl_window))
                history.append((ep + 1, avg_r, avg_pnl, last_loss))
                print(
                    f"ep {ep+1:>7d} | mean_return={avg_r:+.4f} | mean_pnl={avg_pnl:+.2f} | "
                    f"eps={eps:.3f} | buffer={len(self.buffer):>6d} | grad_steps={self.grad_steps:>6d} | "
                    f"last_loss={last_loss} | best_pnl={best_pnl:+.2f}@ep{best_ep}",
                    flush=True,
                )

        return history

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        sd = remap_state_dict(ckpt["q_net"], self.cfg.arch)
        self.q_net.load_state_dict(sd, strict=True)
        self.target_net.load_state_dict(sd, strict=True)
