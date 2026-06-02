"""Environment exposing one TRON-style trader vs. ZI background agents.

Faithful to the paper "A Financial Market Simulation Environment for Trading
Agents Using Deep Reinforcement Learning" (Mascioli et al., ICAIF 2024).
The TRON agent emits factored discrete actions (s_idx, eta_idx) which are
then translated into a ZI-style order (Eq. 5) with the eta-threshold rule
of Section 4.1.
"""

from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

import math
import random

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.private_values.private_values import PrivateValues


DEFAULT_S_GRID: Tuple[float, ...] = tuple(float(i) for i in range(0, 1050, 50))   # 21 values: 0..1000 step 50
DEFAULT_ETA_GRID: Tuple[float, ...] = tuple(round(i / 20.0, 3) for i in range(21))  # 21 values: 0.0..1.0


class TRONEnv(gym.Env):
    """One TRON-style trader vs. a population of zero-intelligence background traders.

    Action: MultiDiscrete([len(s_grid), len(eta_grid)]) -> (s, eta) pair.
    Order is constructed via the ZI rule:

        p_base = f_est + pv_for_exchange(position, side)
        p_offset = p_base - s   if side == BUY
                   p_base + s   if side == SELL

    With eta-threshold: if buy and (p_base - best_ask) > eta * s, post at best_ask;
    if sell and (best_bid - p_base) > eta * s, post at best_bid.

    Observation (10 floats):
        0. time remaining fraction in [0, 1]
        1. fundamental estimate, mean-centered, scaled by price_scale
        2. best bid (scaled, 0 if empty)
        3. best ask (scaled, 0 if empty)
        4. position / q_max in [-1, 1]
        5. private value for next BUY (scaled by pv_scale)
        6. private value for next SELL (scaled by pv_scale)
        7. midprice move (scaled by price_scale)
        8. volume imbalance (clipped tanh)
        9. side indicator: +1 buy, -1 sell

    Reward: change in agent's mark-to-market portfolio value between consecutive
    arrivals (uses estimated fundamental + private values + cash). The final
    step also adds the settlement gap to the realised final fundamental.
    Reward is divided by `reward_scale` for numerical stability.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_background_agents: int = 24,
        sim_time: int = 2_000,
        lam: float = 5e-4,
        mean: float = 1e5,
        kappa: float = 0.01,
        shock_var: float = 1e6,
        q_max: int = 10,
        pv_var: float = 5e6,
        zi_shade: Sequence[float] = (450.0, 540.0),
        zi_eta: float = 1.0,
        s_grid: Sequence[float] = DEFAULT_S_GRID,
        eta_grid: Sequence[float] = DEFAULT_ETA_GRID,
        price_scale: Optional[float] = None,
        pv_scale: Optional[float] = None,
        reward_scale: Optional[float] = None,
    ):
        super().__init__()

        self.num_bg = int(num_background_agents)
        self.self_id = self.num_bg
        self.sim_time = int(sim_time)
        self.lam = float(lam)
        self.mean = float(mean)
        self.kappa = float(kappa)
        self.shock_var = float(shock_var)
        self.q_max = int(q_max)
        self.pv_var = float(pv_var)
        self.zi_shade = list(zi_shade)
        self.zi_eta = float(zi_eta)

        self.s_grid = np.asarray(s_grid, dtype=np.float32)
        self.eta_grid = np.asarray(eta_grid, dtype=np.float32)

        # Normalisers
        self.price_scale = float(price_scale) if price_scale is not None else math.sqrt(self.shock_var / max(self.kappa, 1e-6))
        self.pv_scale = float(pv_scale) if pv_scale is not None else math.sqrt(self.pv_var)
        self.reward_scale = float(reward_scale) if reward_scale is not None else self.price_scale

        # Runtime state
        self.market: Optional[Market] = None
        self.agents: Dict[int, ZIAgent] = {}
        self.self_pv: Optional[PrivateValues] = None
        self.position = 0
        self.cash = 0.0
        self.last_mark = 0.0
        self.time = 0
        self.side: int = BUY
        self.midprice_history: list = []
        self._order_counter = 0

        self.arrivals_bg: Dict[int, list] = defaultdict(list)
        self.arrivals_self: Dict[int, list] = defaultdict(list)

        # Spaces
        self.action_space = spaces.MultiDiscrete([len(self.s_grid), len(self.eta_grid)])
        obs_low = np.array([0.0, -10.0, -10.0, -10.0, -1.0, -10.0, -10.0, -10.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # ---------- arrival scheduling ----------

    def _geo_gap(self, p: float) -> int:
        return int(self.np_random.geometric(p))

    def _schedule_next_self(self, from_time: int) -> None:
        self.arrivals_self[from_time + self._geo_gap(self.lam)].append(self.self_id)

    def _schedule_next_bg(self, agent_id: int, from_time: int) -> None:
        self.arrivals_bg[from_time + self._geo_gap(self.lam)].append(agent_id)

    # ---------- market helpers ----------

    def _new_market(self) -> Market:
        fund = LazyGaussianMeanReverting(
            mean=self.mean,
            final_time=self.sim_time + 1,
            r=self.kappa,
            shock_var=self.shock_var,
        )
        return Market(fundamental=fund, time_steps=self.sim_time)

    def _estimate_fundamental(self) -> float:
        mean, kappa, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()
        rho = (1.0 - kappa) ** (T - t)
        return (1.0 - rho) * mean + rho * val

    def _mark_to_market(self) -> float:
        est = self._estimate_fundamental()
        return (
            self.position * est
            + self.cash
            + self.self_pv.value_at_position(self.position)
        )

    def _midprice_or_estimate(self) -> float:
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()
        if math.isinf(best_bid) or math.isinf(best_ask):
            return self._estimate_fundamental()
        return 0.5 * (best_bid + best_ask)

    # ---------- obs/action ----------

    def _next_order_id(self) -> int:
        self._order_counter += 1
        return self.self_id * 1_000_000 + self._order_counter

    def _build_observation(self) -> np.ndarray:
        est = self._estimate_fundamental()
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()
        bid_obs = 0.0 if math.isinf(best_bid) else (best_bid - self.mean) / self.price_scale
        ask_obs = 0.0 if math.isinf(best_ask) else (best_ask - self.mean) / self.price_scale

        pv_buy = self.self_pv.value_for_exchange(self.position, BUY) / self.pv_scale
        pv_sell = self.self_pv.value_for_exchange(self.position, SELL) / self.pv_scale

        mid = self._midprice_or_estimate()
        self.midprice_history.append(mid)
        if len(self.midprice_history) >= 2:
            midprice_move = (mid - self.midprice_history[-2]) / self.price_scale
        else:
            midprice_move = 0.0

        # Volume imbalance (queue): (#buy - #sell) / (#buy + #sell) within 5-tick window.
        ob = self.market.order_book
        buy_q = len(ob.buy_unmatched.heap) if hasattr(ob.buy_unmatched, "heap") else 0
        sell_q = len(ob.sell_unmatched.heap) if hasattr(ob.sell_unmatched, "heap") else 0
        denom = buy_q + sell_q
        vol_imb = (buy_q - sell_q) / denom if denom > 0 else 0.0

        side_ind = 1.0 if self.side == BUY else -1.0

        obs = np.array(
            [
                max(0.0, (self.sim_time - self.time) / self.sim_time),
                (est - self.mean) / self.price_scale,
                bid_obs,
                ask_obs,
                np.clip(self.position / self.q_max, -1.0, 1.0),
                pv_buy,
                pv_sell,
                np.clip(midprice_move, -10.0, 10.0),
                np.clip(vol_imb, -1.0, 1.0),
                side_ind,
            ],
            dtype=np.float32,
        )
        return obs

    def _build_self_order(self, action: Sequence[int]) -> Order:
        s = float(self.s_grid[int(action[0])])
        eta = float(self.eta_grid[int(action[1])])
        est = self._estimate_fundamental()
        pv_val = self.self_pv.value_for_exchange(self.position, self.side)
        base = est + pv_val
        if self.side == BUY:
            price = base - s
            best_ask = self.market.order_book.get_best_ask()
            if not math.isinf(best_ask) and (base - best_ask) > eta * s:
                price = best_ask
        else:  # SELL
            price = base + s
            best_bid = self.market.order_book.get_best_bid()
            if not math.isinf(best_bid) and (best_bid - base) > eta * s:
                price = best_bid

        return Order(
            price=float(price),
            quantity=1.0,
            agent_id=self.self_id,
            time=self.market.get_time(),
            order_type=self.side,
            order_id=self._next_order_id(),
        )

    # ---------- gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.time = 0
        self._order_counter = 0
        self.position = 0
        self.cash = 0.0
        self.self_pv = PrivateValues(self.q_max, self.pv_var)
        self.midprice_history = []

        self.market = self._new_market()
        self.agents = {
            aid: ZIAgent(
                agent_id=aid,
                market=self.market,
                q_max=self.q_max,
                shade=list(self.zi_shade),
                pv_var=self.pv_var,
                eta=self.zi_eta,
            )
            for aid in range(self.num_bg)
        }

        self.arrivals_bg = defaultdict(list)
        self.arrivals_self = defaultdict(list)
        for aid in range(self.num_bg):
            self._schedule_next_bg(aid, from_time=0)
        self._schedule_next_self(from_time=0)

        # Side for first TRON arrival
        self.side = BUY if self.np_random.random() < 0.5 else SELL

        self._advance_until_self_arrival()
        self.last_mark = self._mark_to_market()
        return self._build_observation(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.time >= self.sim_time:
            return self._build_observation(), 0.0, True, False, {}

        # Post order
        self.market.event_queue.set_time(self.time)
        self.market.withdraw_all(self.self_id)
        self.market.add_orders([self._build_self_order(action)])

        # Background agents arriving now act
        self._background_agents_act(self.time)
        # Clear market this tick
        self._clear_and_settle()

        # Advance and schedule next self-arrival
        self.time += 1
        self._schedule_next_self(from_time=self.time - 1)

        # Run until next self-arrival or sim end
        self._advance_until_self_arrival()

        mtm = self._mark_to_market()
        reward = (mtm - self.last_mark) / self.reward_scale
        self.last_mark = mtm

        terminated = self.time >= self.sim_time
        if terminated:
            fund_final = self.market.get_final_fundamental()
            terminal_val = (
                self.position * fund_final
                + self.cash
                + self.self_pv.value_at_position(self.position)
            )
            reward += (terminal_val - mtm) / self.reward_scale
            self.last_mark = terminal_val

        # Next side (fresh coin flip per arrival, like ZI)
        self.side = BUY if self.np_random.random() < 0.5 else SELL
        obs = self._build_observation()
        return obs, float(reward), terminated, False, {}

    # ---------- internals ----------

    def _background_agents_act(self, t: int) -> None:
        agent_ids = self.arrivals_bg.pop(t, [])
        if not agent_ids:
            return
        self.market.event_queue.set_time(t)
        for aid in agent_ids:
            self.market.withdraw_all(aid)
            self.market.add_orders(self.agents[aid].take_action())
            self._schedule_next_bg(aid, from_time=t)

    def _clear_and_settle(self) -> None:
        new_orders = self.market.step()
        for matched in new_orders:
            quantity = matched.order.order_type * matched.order.quantity
            cash = -matched.price * matched.order.quantity * matched.order.order_type
            if matched.order.agent_id == self.self_id:
                self.position += int(quantity)
                self.cash += float(cash)
            else:
                self.agents[matched.order.agent_id].update_position(quantity, cash)

    def _advance_until_self_arrival(self) -> None:
        while self.time < self.sim_time and self.self_id not in self.arrivals_self.get(self.time, ()):
            self._background_agents_act(self.time)
            self._clear_and_settle()
            self.time += 1
        self.arrivals_self.pop(self.time, None)

    def get_terminal_pnl(self) -> float:
        """Profit at end of sim relative to settlement on the realised final fundamental."""
        fund_final = self.market.get_final_fundamental()
        return (
            self.position * fund_final
            + self.cash
            + self.self_pv.value_at_position(self.position)
        )
