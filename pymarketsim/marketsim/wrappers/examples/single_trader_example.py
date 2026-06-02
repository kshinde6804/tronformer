"""Random-policy rollout for the SingleTraderEnv RL wrapper."""

import numpy as np
import gymnasium as gym

import marketsim.wrappers  # registers PyMarketSim-SingleTrader-v0


def main(num_steps: int = 200, seed: int = 0) -> None:
    env = gym.make(
        "PyMarketSim-SingleTrader-v0",
        num_background_agents=25,
        sim_time=1_000,
    )
    obs, info = env.reset(seed=seed)
    print("obs0:", np.round(obs, 4))

    total_reward = 0.0
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            print(f"episode finished at step {step + 1}, return={total_reward:.4f}")
            total_reward = 0.0
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
