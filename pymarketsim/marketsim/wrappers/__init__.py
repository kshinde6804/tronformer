from gymnasium.envs.registration import register

from marketsim.wrappers.trader_env import SingleTraderEnv
from marketsim.wrappers.tron_env import TRONEnv

register(
    id="PyMarketSim-SingleTrader-v0",
    entry_point="marketsim.wrappers.trader_env:SingleTraderEnv",
    max_episode_steps=None,
)

register(
    id="PyMarketSim-TRON-v0",
    entry_point="marketsim.wrappers.tron_env:TRONEnv",
    max_episode_steps=None,
)

__all__ = ["SingleTraderEnv", "TRONEnv"]
