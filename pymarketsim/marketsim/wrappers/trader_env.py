"""A clean Gymnasium environment exposing one RL-controlled trader.

The RL agent participates in a continuous-double-auction limit order book
populated by zero-intelligence (ZI) background traders. On each `step`, the
RL agent submits one order (price + side); the simulator advances the market
until the next RL arrival, executing background trades in the meantime.

Designed to be drop-in usable with standard RL libraries (Stable-Baselines3,
CleanRL, etc.) — observations and actions are pre-normalised, the action
space is `Box(-1, 1)`, and reward is a per-step PnL delta.
"""

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from marketsim.private_values.private_values import PrivateValues


class SingleTraderEnv(gym.Env):
    """Single RL trader vs. a population of zero-intelligence background agents.

    Observation (shape (7,), values in [-1, 1] except where noted):
        0. fraction of simulation time remaining
        1. fundamental estimate, mean-centered and scaled by `price_scale`
        2. best bid (mean-centered, scaled); 0.0 if book is empty
        3. best ask (mean-centered, scaled); 0.0 if book is empty
        4. inventory / q_max  (clipped to [-1, 1])
        5. marginal private value for the NEXT BUY,  scaled by `pv_scale`
        6. marginal private value for the NEXT SELL, scaled by `pv_scale`

    Action (shape (2,), Box(-1, 1)):
        action[0]  side score    — buy if >= 0 else sell
        action[1]  shade score   — 0 = aggressive (at the fundamental + pv);
                                   1 = passive by `max_shade` ticks below ask
                                   (or above bid for a sell). Negative values
                                   place the order through the touch (i.e.
                                   marketable) by that much.

        Concretely the submitted price is::

            base = fundamental_estimate + private_value_at(position, side)
            offset = action[1] * max_shade
            price = base - offset    if side == BUY
                    base + offset    if side == SELL

    Reward:
        Change in mark-to-market portfolio value between consecutive RL
        arrivals, divided by `reward_scale`. Mark-to-market uses the latest
        fundamental estimate plus realised private value at the current
        position. The terminal step adds the final fundamental value.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_background_agents: int = 25,
        sim_time: int = 1_000,
        lam: float = 0.075,
        lam_self: float = 0.05,
        mean: float = 1e5,
        r: float = 0.05,
        shock_var: float = 5e6,
        q_max: int = 10,
        pv_var: float = 5e6,
        zi_shade=(250.0, 500.0),
        max_shade: float = 500.0,
        price_scale: Optional[float] = None,
        pv_scale: Optional[float] = None,
        reward_scale: Optional[float] = None,
    ):
        super().__init__()

        self.num_background_agents = int(num_background_agents)
        self.self_agent_id = self.num_background_agents
        self.sim_time = int(sim_time)
        self.lam = float(lam)
        self.lam_self = float(lam_self)

        # Fundamental params
        self.mean = float(mean)
        self.r = float(r)
        self.shock_var = float(shock_var)

        # Trader params
        self.q_max = int(q_max)
        self.pv_var = float(pv_var)
        self.zi_shade = list(zi_shade)
        self.max_shade = float(max_shade)

        # Normalisers — picked so observations land near unit scale.
        self.price_scale = float(price_scale) if price_scale is not None else math.sqrt(self.shock_var / max(self.r, 1e-6))
        self.pv_scale = float(pv_scale) if pv_scale is not None else math.sqrt(self.pv_var)
        self.reward_scale = float(reward_scale) if reward_scale is not None else self.price_scale

        # Market + background agents are created in reset() so that seeding is honored.
        self.market: Optional[Market] = None
        self.agents: Dict[int, ZIAgent] = {}
        self.self_pv: Optional[PrivateValues] = None
        self.self_position = 0
        self.self_cash = 0.0
        self.last_mark_to_market = 0.0

        # Arrival schedules
        self.arrivals_bg: Dict[int, list] = defaultdict(list)
        self.arrivals_self: Dict[int, list] = defaultdict(list)
        self.time = 0

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, -10.0, -10.0, -1.0, -10.0, -10.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Bookkeeping
        self._order_counter = 0

    # ---------- helpers ----------

    def _sample_geometric_gap(self, p: float) -> int:
        # gap >= 1 between arrivals; matches the spirit of the existing wrappers
        # (geometric on success-probability p).
        return int(self.np_random.geometric(p))

    def _schedule_next_self(self, from_time: int) -> None:
        gap = self._sample_geometric_gap(self.lam_self)
        self.arrivals_self[from_time + gap].append(self.self_agent_id)

    def _schedule_next_bg(self, agent_id: int, from_time: int) -> None:
        gap = self._sample_geometric_gap(self.lam)
        self.arrivals_bg[from_time + gap].append(agent_id)

    def _new_market(self) -> Market:
        fundamental = LazyGaussianMeanReverting(
            mean=self.mean,
            final_time=self.sim_time + 1,
            r=self.r,
            shock_var=self.shock_var,
        )
        return Market(fundamental=fundamental, time_steps=self.sim_time)

    def _estimate_fundamental(self) -> float:
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()
        rho = (1.0 - r) ** (T - t)
        return (1.0 - rho) * mean + rho * val

    def _mark_to_market(self) -> float:
        est = self._estimate_fundamental()
        return (
            self.self_position * est
            + self.self_cash
            + self.self_pv.value_at_position(self.self_position)
        )

    def _build_observation(self) -> np.ndarray:
        est = self._estimate_fundamental()
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()

        bid_obs = 0.0 if math.isinf(best_bid) else (best_bid - self.mean) / self.price_scale
        ask_obs = 0.0 if math.isinf(best_ask) else (best_ask - self.mean) / self.price_scale

        pv_buy = self.self_pv.value_for_exchange(self.self_position, BUY) / self.pv_scale
        pv_sell = self.self_pv.value_for_exchange(self.self_position, SELL) / self.pv_scale

        obs = np.array(
            [
                max(0.0, (self.sim_time - self.time) / self.sim_time),
                (est - self.mean) / self.price_scale,
                bid_obs,
                ask_obs,
                np.clip(self.self_position / self.q_max, -1.0, 1.0),
                pv_buy,
                pv_sell,
            ],
            dtype=np.float32,
        )
        return obs

    def _next_order_id(self) -> int:
        self._order_counter += 1
        return self.self_agent_id * 1_000_000 + self._order_counter

    def _build_self_order(self, action: np.ndarray) -> Order:
        side = BUY if action[0] >= 0.0 else SELL
        shade = float(np.clip(action[1], -1.0, 1.0)) * self.max_shade
        base = self._estimate_fundamental() + self.self_pv.value_for_exchange(self.self_position, side)
        price = base - shade if side == BUY else base + shade
        return Order(
            price=float(price),
            quantity=1.0,
            agent_id=self.self_agent_id,
            time=self.market.get_time(),
            order_type=side,
            order_id=self._next_order_id(),
        )

    # ---------- gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # The marketsim internals use python `random` and `numpy.random` directly
        # rather than our np_random, so we re-seed those too for reproducibility.
        if seed is not None:
            np.random.seed(seed)
            import random as _random
            _random.seed(seed)

        self.time = 0
        self._order_counter = 0
        self.self_position = 0
        self.self_cash = 0.0
        self.self_pv = PrivateValues(self.q_max, self.pv_var)

        self.market = self._new_market()

        self.agents = {}
        for agent_id in range(self.num_background_agents):
            self.agents[agent_id] = ZIAgent(
                agent_id=agent_id,
                market=self.market,
                q_max=self.q_max,
                shade=list(self.zi_shade),
                pv_var=self.pv_var,
            )

        self.arrivals_bg = defaultdict(list)
        self.arrivals_self = defaultdict(list)
        for agent_id in range(self.num_background_agents):
            self._schedule_next_bg(agent_id, from_time=0)
        self._schedule_next_self(from_time=0)

        # Fast-forward until the RL agent first arrives.
        self._advance_until_self_arrival()
        self.last_mark_to_market = self._mark_to_market()
        return self._build_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.time >= self.sim_time:
            return self._terminal_step()

        # 1. RL trader posts order (replacing its previous orders).
        self.market.event_queue.set_time(self.time)
        self.market.withdraw_all(self.self_agent_id)
        order = self._build_self_order(np.asarray(action, dtype=np.float32))
        self.market.add_orders([order])

        # 2. Background agents at this timestep act.
        self._background_agents_act(self.time)

        # 3. Clear the market for this timestep.
        self._clear_and_settle()

        # 4. Schedule the RL agent's next arrival.
        self.time += 1
        self._schedule_next_self(from_time=self.time - 1)

        # 5. Run background-only steps until next RL arrival or end of sim.
        truncated = self._advance_until_self_arrival()

        # 6. Compute reward as PnL delta.
        mtm = self._mark_to_market()
        reward = (mtm - self.last_mark_to_market) / self.reward_scale
        self.last_mark_to_market = mtm

        terminated = self.time >= self.sim_time
        if terminated:
            # Settle on the realised final fundamental rather than the estimate.
            fund_final = self.market.get_final_fundamental()
            terminal_value = (
                self.self_position * fund_final
                + self.self_cash
                + self.self_pv.value_at_position(self.self_position)
            )
            reward += (terminal_value - mtm) / self.reward_scale
            self.last_mark_to_market = terminal_value

        obs = self._build_observation()
        return obs, float(reward), terminated, False, {}

    # ---------- internals ----------

    def _terminal_step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self._build_observation(), 0.0, True, False, {}

    def _background_agents_act(self, t: int) -> None:
        agent_ids = self.arrivals_bg.pop(t, [])
        if not agent_ids:
            return
        self.market.event_queue.set_time(t)
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            self.market.withdraw_all(agent_id)
            self.market.add_orders(agent.take_action())
            self._schedule_next_bg(agent_id, from_time=t)

    def _clear_and_settle(self) -> None:
        new_orders = self.market.step()
        for matched in new_orders:
            quantity = matched.order.order_type * matched.order.quantity
            cash = -matched.price * matched.order.quantity * matched.order.order_type
            if matched.order.agent_id == self.self_agent_id:
                self.self_position += int(quantity)
                self.self_cash += float(cash)
            else:
                self.agents[matched.order.agent_id].update_position(quantity, cash)

    def _advance_until_self_arrival(self) -> bool:
        """Advance time until the RL agent arrives or the sim ends.

        Returns True if the sim ended before the RL agent arrived again
        (which is mapped to `truncated=False, terminated=True` by `step`).
        """
        while self.time < self.sim_time and self.self_agent_id not in self.arrivals_self.get(self.time, ()):
            self._background_agents_act(self.time)
            self._clear_and_settle()
            self.time += 1
        # Consume the self-arrival marker (so we don't double-process if step is
        # called again without a new schedule).
        self.arrivals_self.pop(self.time, None)
        return False

    # ---------- convenience ----------

    def render(self) -> None:  # pragma: no cover - no rendering implemented
        return None

    def close(self) -> None:  # pragma: no cover
        return None
