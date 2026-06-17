"""TRON networks: dueling DQN with either LSTM or causal Transformer recurrence.

The LSTM variant (`TRONNet`) matches the architecture from Mascioli et al.,
ICAIF 2024, Fig 3. The Transformer variant (`TRONTransformerNet`) keeps the
input projection and dueling heads identical but replaces the LSTM with a
small causal Transformer encoder.

Both networks have the same `forward(obs, hidden)` signature returning
`(q_s, q_eta, new_hidden)`:

  - During *training*, `obs` is `(B, T, in_dim)` (full episode batches) and
    `hidden` is `None`; we forward causally and return per-step Q-values.
  - During *inference* (sequential acting), `obs` is `(B, 1, in_dim)` and
    `hidden` carries the recurrent state across steps:
      - LSTM:        `(h, c)` tuple, each `(1, B, hidden_dim)`
      - Transformer: cached per-step projected embeddings `(B, T_past, d_model)`

The dueling heads output Q-values for 21 `s` choices and 21 `eta` choices
independently:  Q_s = V_s + (A_s - mean(A_s)); same for eta.

## State-dict compatibility

Old checkpoints (pre-refactor) used flat attribute names. `remap_state_dict`
performs a key remap so old `.pt` files load into the refactored classes.
"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn


class InputProjection(nn.Module):
    """Shared Linear→ReLU input projection used by both TRON variants."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactoredDuelingHeads(nn.Module):
    """Shared dueling heads for the factored (s, eta) action space."""

    def __init__(self, hidden_dim: int, n_s: int, n_eta: int) -> None:
        super().__init__()
        self.n_s = n_s
        self.n_eta = n_eta
        # Advantage head: outputs n_s + n_eta values.
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_s + n_eta),
        )
        # Value head: one scalar per factored action group (2 total).
        self.val_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q_s, q_eta) from recurrent output x."""
        adv = self.adv_head(x)                    # (..., n_s + n_eta)
        val = self.val_head(x)                    # (..., 2)
        adv_s, adv_eta = adv.split([self.n_s, self.n_eta], dim=-1)
        val_s = val[..., 0:1]
        val_eta = val[..., 1:2]
        q_s = val_s + (adv_s - adv_s.mean(dim=-1, keepdim=True))
        q_eta = val_eta + (adv_eta - adv_eta.mean(dim=-1, keepdim=True))
        return q_s, q_eta


class TRONNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_s: int = 21,
        n_eta: int = 21,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_s = n_s
        self.n_eta = n_eta
        self.hidden_dim = hidden_dim

        self.input_proj = InputProjection(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.heads = FactoredDuelingHeads(hidden_dim, n_s, n_eta)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run a forward pass over a (batch, seq, input_dim) tensor.

        Returns:
            q_s:    (batch, seq, n_s)   Q-values for the `s` action factor
            q_eta:  (batch, seq, n_eta) Q-values for the `eta` action factor
            hidden: final LSTM hidden state (h, c), each (1, batch, hidden_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = self.input_proj(obs)
        out, hidden = self.lstm(x, hidden)
        q_s, q_eta = self.heads(out)
        return q_s, q_eta, hidden

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h, c


def _sinusoidal_pos_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class TRONTransformerNet(nn.Module):
    """Dueling DQN with a causal Transformer encoder in place of the LSTM."""

    def __init__(
        self,
        input_dim: int,
        n_s: int = 21,
        n_eta: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_seq: int = 64,
    ) -> None:
        super().__init__()
        self.n_s = n_s
        self.n_eta = n_eta
        self.d_model = d_model
        self.max_seq = max_seq

        self.input_proj = InputProjection(input_dim, d_model)
        # Sinusoidal positional encoding, registered as buffer so it moves with .to(device).
        self.register_buffer("pos_enc", _sinusoidal_pos_encoding(max_seq, d_model), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for small RL transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.heads = FactoredDuelingHeads(d_model, n_s, n_eta)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal forward.

        Args:
            obs: (B, T_new, input_dim) — new observations to append
            hidden: (B, T_past, d_model) — cached pre-positional embeddings,
                or None to start from an empty context

        Returns:
            q_s: (B, T_total, n_s) Q-values for the `s` action factor
            q_eta: (B, T_total, n_eta)
            new_hidden: (B, T_total, d_model) — pre-positional cache for next call
                        (truncated from the left if it exceeds max_seq)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        x_new = self.input_proj(obs)  # (B, T_new, d_model)
        if hidden is not None and hidden.size(1) > 0:
            x_cache = torch.cat([hidden, x_new], dim=1)
        else:
            x_cache = x_new

        if x_cache.size(1) > self.max_seq:
            x_cache = x_cache[:, -self.max_seq:]

        T = x_cache.size(1)
        x_pe = x_cache + self.pos_enc[:T].unsqueeze(0)

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x_pe.device),
            diagonal=1,
        )
        out = self.encoder(x_pe, mask=causal_mask, is_causal=True)  # (B, T, d_model)

        q_s, q_eta = self.heads(out)

        return q_s, q_eta, x_cache  # x_cache is the un-positional cache

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, 0, self.d_model, device=device)


# ---------------------------------------------------------------------------
# Key remap for backward compatibility with pre-refactor checkpoints
# ---------------------------------------------------------------------------

# Old key → new key mappings for TRONNet (LSTM).
# Pre-refactor: encoder.0.weight → input_proj.net.0.weight
#               adv_head.* → heads.adv_head.*
#               val_head.* → heads.val_head.*
_LSTM_KEY_MAP = [
    ("encoder.0.", "input_proj.net.0."),
    ("encoder.2.", "input_proj.net.2."),  # (unused; encoder had no index 2, but safe)
    ("adv_head.", "heads.adv_head."),
    ("val_head.", "heads.val_head."),
]

# Old key → new key mappings for TRONTransformerNet.
# Pre-refactor: input_proj.0.* → input_proj.net.0.*
#               adv_head.* → heads.adv_head.*
#               val_head.* → heads.val_head.*
_XFMR_KEY_MAP = [
    ("input_proj.0.", "input_proj.net.0."),
    ("input_proj.2.", "input_proj.net.2."),  # (unused; same reason)
    ("adv_head.", "heads.adv_head."),
    ("val_head.", "heads.val_head."),
]


def remap_state_dict(state_dict: dict, arch: str) -> dict:
    """Remap old (pre-refactor) checkpoint keys to the current layout."""
    key_map = _LSTM_KEY_MAP if arch == "lstm" else _XFMR_KEY_MAP
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in key_map:
            if new_k.startswith(old_prefix):
                new_k = new_prefix + new_k[len(old_prefix):]
                break
        new_sd[new_k] = v
    return new_sd


def build_network(
    arch: str,
    input_dim: int,
    n_s: int = 21,
    n_eta: int = 21,
    **kwargs,
) -> Union[TRONNet, TRONTransformerNet]:
    arch = arch.lower()
    if arch == "lstm":
        return TRONNet(input_dim=input_dim, n_s=n_s, n_eta=n_eta, **kwargs)
    if arch in ("transformer", "tron-transformer", "xfmr"):
        return TRONTransformerNet(input_dim=input_dim, n_s=n_s, n_eta=n_eta, **kwargs)
    raise ValueError(f"Unknown arch: {arch!r}")
