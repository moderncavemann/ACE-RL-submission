"""Configuration required by the included order-book environment."""

from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    lob_depth: int = 3
    tick_size: float = 0.01
    max_price_offset: int = 10
    max_inventory: int = 100
    inventory_penalty: float = 0.001
    reward_scale: float = 0.1
    episode_length: int = 1000
    ou_theta: float = 0.1
    ou_mu: float = 100.0
    ou_sigma: float = 0.5
    noise_trader_intensity: float = 0.3
    value_investor_fraction: float = 0.2

    @property
    def state_dim(self) -> int:
        return 4 * self.lob_depth + 2

    @property
    def action_dim(self) -> int:
        return 2 * self.max_price_offset + 2
