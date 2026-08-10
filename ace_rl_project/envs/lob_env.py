"""
=============================================================================
High-Fidelity Limit Order Book Environment
Implements the LOB simulation from Section 4.1 of the ACE-RL paper.

Key Features:
- Price-time priority matching
- Ornstein-Uhlenbeck fundamental price process
- Background traders (noise, momentum, value investors)
- Multi-agent support for N market makers

MDP Specification:
- State: s_t = [P_bid^{1:k}, V_bid^{1:k}, P_ask^{1:k}, V_ask^{1:k}, I_t, τ_t]
- Action: Discrete price offsets {-Kδ, ..., +Kδ, ∅}
- Reward: R_t = ΔW_t - λ(I_t)²

PERFORMANCE NOTE:
-----------------
Current implementation uses pure Python (heapq + list iteration).
For paper experiments (500-1000 episodes, 1000 steps each), this is sufficient.

For large-scale training (1M+ steps), consider:
1. Numba JIT compilation for match_orders() - ~10-50x speedup
2. numpy array-based LOB (fixed Top-K levels only) - simpler & faster
3. Cython extension for order matching loop

Example Numba acceleration:
```python
from numba import jit

@jit(nopython=True)
def fast_match(bids, asks, ...):
    ...
```
=============================================================================
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import heapq

import sys
sys.path.append('..')
from experiments.config import EnvironmentConfig


@dataclass
class Order:
    """Limit order representation."""
    order_id: int
    agent_id: int           # -1 for background traders
    side: str               # 'bid' or 'ask'
    price: float
    quantity: int
    timestamp: int

    def __lt__(self, other):
        """For heap ordering: price-time priority."""
        if self.side == 'bid':
            # Higher price, then earlier time
            return (-self.price, self.timestamp) < (-other.price, other.timestamp)
        else:
            # Lower price, then earlier time
            return (self.price, self.timestamp) < (other.price, other.timestamp)


class LimitOrderBook:
    """
    Price-time priority limit order book.
    Implements standard exchange matching rules.
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: List[Order] = []  # Max-heap (negate prices)
        self.asks: List[Order] = []  # Min-heap
        self.order_counter = 0
        self.trades: List[Dict] = []

    def add_order(self, agent_id: int, side: str, price: float,
                  quantity: int, timestamp: int) -> int:
        """Add a limit order to the book."""
        self.order_counter += 1
        order = Order(
            order_id=self.order_counter,
            agent_id=agent_id,
            side=side,
            price=self._round_price(price),
            quantity=quantity,
            timestamp=timestamp
        )

        if side == 'bid':
            heapq.heappush(self.bids, order)
        else:
            heapq.heappush(self.asks, order)

        return order.order_id

    def _round_price(self, price: float) -> float:
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size

    def match_orders(self, timestamp: int) -> List[Dict]:
        """
        Execute matching at current time.
        Returns list of executed trades.
        """
        trades = []

        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            # Check if orders can match
            if best_bid.price >= best_ask.price:
                # Execute at the resting order's price (price-time priority)
                exec_price = best_ask.price if best_ask.timestamp < best_bid.timestamp else best_bid.price
                exec_qty = min(best_bid.quantity, best_ask.quantity)

                trade = {
                    'timestamp': timestamp,
                    'price': exec_price,
                    'quantity': exec_qty,
                    'buyer_id': best_bid.agent_id,
                    'seller_id': best_ask.agent_id,
                    'buyer_order_id': best_bid.order_id,
                    'seller_order_id': best_ask.order_id,
                }
                trades.append(trade)
                self.trades.append(trade)

                # Update quantities
                best_bid.quantity -= exec_qty
                best_ask.quantity -= exec_qty

                # Remove filled orders
                if best_bid.quantity == 0:
                    heapq.heappop(self.bids)
                if best_ask.quantity == 0:
                    heapq.heappop(self.asks)
            else:
                break

        return trades

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        while self.bids and self.bids[0].quantity == 0:
            heapq.heappop(self.bids)
        return self.bids[0].price if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        while self.asks and self.asks[0].quantity == 0:
            heapq.heappop(self.asks)
        return self.asks[0].price if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        """Get mid-price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_total_volume(self) -> float:
        """Get total volume in order book (both sides)."""
        bid_volume = sum(o.quantity for o in self.bids if o.quantity > 0)
        ask_volume = sum(o.quantity for o in self.asks if o.quantity > 0)
        return bid_volume + ask_volume

    def get_order_imbalance(self) -> float:
        """
        Get order flow imbalance (for market factors).

        Returns:
            Imbalance in [-1, 1]: positive = more bids, negative = more asks
        """
        bid_volume = sum(o.quantity for o in self.bids if o.quantity > 0)
        ask_volume = sum(o.quantity for o in self.asks if o.quantity > 0)
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    def get_depth(self, side: str, levels: int) -> Tuple[List[float], List[int]]:
        """
        Get prices and volumes for top-k levels.

        Returns:
            prices: List of prices (length = levels)
            volumes: List of volumes (length = levels)
        """
        if side == 'bid':
            orders = sorted([o for o in self.bids if o.quantity > 0],
                          key=lambda x: (-x.price, x.timestamp))
        else:
            orders = sorted([o for o in self.asks if o.quantity > 0],
                          key=lambda x: (x.price, x.timestamp))

        # Aggregate by price level
        price_levels: Dict[float, int] = {}
        for order in orders:
            if order.price not in price_levels:
                price_levels[order.price] = 0
            price_levels[order.price] += order.quantity

        # Sort and extract top-k
        if side == 'bid':
            sorted_levels = sorted(price_levels.items(), key=lambda x: -x[0])
        else:
            sorted_levels = sorted(price_levels.items(), key=lambda x: x[0])

        prices = [p for p, v in sorted_levels[:levels]]
        volumes = [v for p, v in sorted_levels[:levels]]

        # Pad if fewer than k levels
        while len(prices) < levels:
            prices.append(0.0)
            volumes.append(0)

        return prices, volumes

    def cancel_agent_orders(self, agent_id: int):
        """Cancel all orders from a specific agent."""
        self.bids = [o for o in self.bids if o.agent_id != agent_id]
        self.asks = [o for o in self.asks if o.agent_id != agent_id]
        heapq.heapify(self.bids)
        heapq.heapify(self.asks)

    def clear(self):
        """Clear all orders."""
        self.bids = []
        self.asks = []
        self.trades = []


class OUProcess:
    """
    Ornstein-Uhlenbeck process for fundamental price.
    dX_t = θ(μ - X_t)dt + σdW_t
    """

    def __init__(self, theta: float = 0.1, mu: float = 100.0,
                 sigma: float = 0.5, x0: Optional[float] = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x = x0 if x0 is not None else mu

    def step(self, dt: float = 1.0) -> float:
        """Advance process by dt."""
        dx = self.theta * (self.mu - self.x) * dt
        dx += self.sigma * np.sqrt(dt) * np.random.randn()
        self.x += dx
        return self.x

    def reset(self, x0: Optional[float] = None):
        """Reset to initial value."""
        self.x = x0 if x0 is not None else self.mu


class BackgroundTrader:
    """
    Background liquidity provider.
    Includes noise traders and value investors.
    """

    def __init__(self, intensity: float = 0.3, value_fraction: float = 0.2):
        self.intensity = intensity
        self.value_fraction = value_fraction

    def generate_orders(self, fundamental: float, mid_price: float,
                       tick_size: float, timestamp: int) -> List[Tuple]:
        """
        Generate background orders for this timestep.

        Returns:
            List of (side, price, quantity) tuples
        """
        orders = []

        # Poisson arrivals
        n_orders = np.random.poisson(self.intensity)

        for _ in range(n_orders):
            if np.random.random() < self.value_fraction:
                # Value investor: trade towards fundamental
                if fundamental > mid_price:
                    side = 'bid'
                    offset = np.random.randint(0, 3)
                    price = mid_price + offset * tick_size
                else:
                    side = 'ask'
                    offset = np.random.randint(0, 3)
                    price = mid_price - offset * tick_size
            else:
                # Noise trader: random side and price
                side = 'bid' if np.random.random() < 0.5 else 'ask'
                offset = np.random.randint(-5, 6)
                price = mid_price + offset * tick_size

            quantity = np.random.randint(1, 5)
            orders.append((side, price, quantity))

        return orders


class LOBEnv(gym.Env):
    """
    Gymnasium environment for multi-agent market making in a LOB.

    Implements the MDP from Section 4.1:
    - State: s_t = [P_bid^{1:k}, V_bid^{1:k}, P_ask^{1:k}, V_ask^{1:k}, I_t, τ_t]
    - Action: Discrete price offsets from mid-price
    - Reward: R_t = ΔW_t - λ(I_t)²
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[EnvironmentConfig] = None,
                 n_agents: int = 3, render_mode: Optional[str] = None):
        super().__init__()

        self.config = config or EnvironmentConfig()
        self.n_agents = n_agents
        self.render_mode = render_mode

        # === LOB ===
        self.lob = LimitOrderBook(tick_size=self.config.tick_size)

        # === Fundamental Price ===
        self.ou_process = OUProcess(
            theta=self.config.ou_theta,
            mu=self.config.ou_mu,
            sigma=self.config.ou_sigma
        )

        # === Background Traders ===
        self.background_trader = BackgroundTrader(
            intensity=self.config.noise_trader_intensity,
            value_fraction=self.config.value_investor_fraction
        )

        # === Agent State ===
        self.inventories = np.zeros(n_agents)
        self.cash = np.zeros(n_agents)
        self.wealth_history: List[np.ndarray] = []

        # === Time ===
        self.current_step = 0
        self.episode_length = self.config.episode_length

        # === Spaces ===
        # State: [P_bid^{1:k}, V_bid^{1:k}, P_ask^{1:k}, V_ask^{1:k}, I_t, τ_t]
        # Normalized to roughly [-1, 1] range
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.state_dim,),
            dtype=np.float32
        )

        # Action: Discrete {-K, ..., -1, 0, +1, ..., +K, HOLD}
        self.action_space = spaces.Discrete(self.config.action_dim)

        # === Metrics tracking ===
        self.price_history: List[float] = []
        self.spread_history: List[float] = []
        self.trade_history: List[Dict] = []

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """
        Construct observation for agent_id.
        s_t = [P_bid^{1:k}, V_bid^{1:k}, P_ask^{1:k}, V_ask^{1:k}, I_t, τ_t]
        """
        k = self.config.lob_depth

        # Get LOB depth
        bid_prices, bid_volumes = self.lob.get_depth('bid', k)
        ask_prices, ask_volumes = self.lob.get_depth('ask', k)

        # Normalize prices relative to mid-price
        mid_price = self.lob.get_mid_price() or self.ou_process.x
        bid_prices_norm = [(p - mid_price) / mid_price if p > 0 else 0 for p in bid_prices]
        ask_prices_norm = [(p - mid_price) / mid_price if p > 0 else 0 for p in ask_prices]

        # Log-normalize volumes
        bid_volumes_norm = [np.log1p(v) / 5.0 for v in bid_volumes]  # Scale factor
        ask_volumes_norm = [np.log1p(v) / 5.0 for v in ask_volumes]

        # Inventory (normalized)
        inventory_norm = self.inventories[agent_id] / self.config.max_inventory

        # Remaining time (normalized)
        tau = (self.episode_length - self.current_step) / self.episode_length

        # Concatenate
        obs = np.array(
            bid_prices_norm + bid_volumes_norm +
            ask_prices_norm + ask_volumes_norm +
            [inventory_norm, tau],
            dtype=np.float32
        )

        return obs

    def _decode_action(self, action: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Decode discrete action to (side, offset).

        Action space: {0, 1, ..., 2K+1}
        - 0 to K-1: bid offsets (-K to -1)
        - K: bid at mid (0)
        - K+1 to 2K: ask offsets (+1 to +K)
        - 2K+1: HOLD (no action)
        """
        K = self.config.max_price_offset

        if action == 2 * K + 1:
            return None, None  # HOLD

        if action <= K:
            # Bid side
            offset = action - K  # Maps 0->-K, K->0
            return 'bid', offset
        else:
            # Ask side
            offset = action - K  # Maps K+1->1, 2K->K
            return 'ask', offset

    def _execute_action(self, agent_id: int, action: int) -> float:
        """
        Execute agent's action and return immediate PnL.
        """
        side, offset = self._decode_action(action)

        if side is None:
            return 0.0  # HOLD

        # Get mid-price
        mid_price = self.lob.get_mid_price()
        if mid_price is None:
            mid_price = self.ou_process.x

        # Calculate limit price
        price = mid_price + offset * self.config.tick_size

        # Check inventory constraints
        if side == 'bid' and self.inventories[agent_id] >= self.config.max_inventory:
            return 0.0  # Cannot buy more
        if side == 'ask' and self.inventories[agent_id] <= -self.config.max_inventory:
            return 0.0  # Cannot sell more

        # Cancel existing orders
        self.lob.cancel_agent_orders(agent_id)

        # Submit new order
        quantity = 1  # Unit size for simplicity
        self.lob.add_order(agent_id, side, price, quantity, self.current_step)

        return 0.0  # PnL calculated after matching

    def _process_trades(self) -> Tuple[Dict[int, float], Dict[int, bool], Dict[int, bool]]:
        """
        Match orders and compute PnL for each agent.

        Returns:
            pnl: Dict mapping agent_id to PnL
            traded: Dict mapping agent_id to whether they traded this step
            traded_with_probe: Dict mapping agent_id to whether they traded with governance probe (agent_id=-2)
        """
        trades = self.lob.match_orders(self.current_step)
        pnl = {i: 0.0 for i in range(self.n_agents)}
        traded = {i: False for i in range(self.n_agents)}
        traded_with_probe = {i: False for i in range(self.n_agents)}

        for trade in trades:
            buyer_id = trade['buyer_id']
            seller_id = trade['seller_id']
            price = trade['price']
            qty = trade['quantity']

            # Update buyer
            if buyer_id >= 0:  # Not background trader
                self.inventories[buyer_id] += qty
                self.cash[buyer_id] -= price * qty
                pnl[buyer_id] -= price * qty
                traded[buyer_id] = True
                # Check if traded with governance probe (agent_id = -2)
                if seller_id == -2:
                    traded_with_probe[buyer_id] = True

            # Update seller
            if seller_id >= 0:  # Not background trader
                self.inventories[seller_id] -= qty
                self.cash[seller_id] += price * qty
                pnl[seller_id] += price * qty
                traded[seller_id] = True
                # Check if traded with governance probe
                if buyer_id == -2:
                    traded_with_probe[seller_id] = True

        self.trade_history.extend(trades)
        self._last_traded = traded  # Store for info dict
        self._last_traded_with_probe = traded_with_probe
        return pnl, traded, traded_with_probe

    def _compute_rewards(self, pnl: Dict[int, float]) -> Dict[int, float]:
        """
        Compute rewards with inventory penalty and scaling.
        R_t = (ΔW_t - λ(I_t)²) * reward_scale

        NOTE: Reward scaling is critical for PPO convergence.
        Raw PnL can be O(100), but PPO prefers rewards in [-10, 10].
        """
        rewards = {}

        for agent_id in range(self.n_agents):
            # Mark-to-market wealth change
            mid_price = self.lob.get_mid_price() or self.ou_process.x
            delta_wealth = pnl[agent_id] + self.inventories[agent_id] * 0  # Simplified

            # Inventory penalty
            inventory_penalty = self.config.inventory_penalty * (self.inventories[agent_id] ** 2)

            # Apply reward scaling for PPO stability
            raw_reward = delta_wealth - inventory_penalty
            rewards[agent_id] = raw_reward * self.config.reward_scale

        return rewards

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset LOB
        self.lob.clear()

        # Reset fundamental price
        self.ou_process.reset()

        # Reset agent state
        self.inventories = np.zeros(self.n_agents)
        self.cash = np.zeros(self.n_agents)
        self.wealth_history = []

        # Reset time
        self.current_step = 0

        # Reset history
        self.price_history = []
        self.spread_history = []
        self.trade_history = []
        self._last_traded = {i: False for i in range(self.n_agents)}
        self._last_traded_with_probe = {i: False for i in range(self.n_agents)}

        # Initialize LOB with some background orders
        mid = self.ou_process.x
        for offset in range(-3, 4):
            if offset < 0:
                self.lob.add_order(-1, 'bid', mid + offset * self.config.tick_size,
                                   np.random.randint(1, 5), 0)
            elif offset > 0:
                self.lob.add_order(-1, 'ask', mid + offset * self.config.tick_size,
                                   np.random.randint(1, 5), 0)

        # Get observations
        observations = {i: self._get_observation(i) for i in range(self.n_agents)}
        infos = {i: {} for i in range(self.n_agents)}

        return observations, infos

    def step(self, actions: Dict[int, int]
             ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict[int, Dict]]:
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            observations, rewards, terminated, truncated, infos
        """
        self.current_step += 1

        # Update fundamental price
        self.ou_process.step()

        # Execute agent actions
        for agent_id, action in actions.items():
            self._execute_action(agent_id, action)

        # Add background orders
        mid_price = self.lob.get_mid_price() or self.ou_process.x
        bg_orders = self.background_trader.generate_orders(
            self.ou_process.x, mid_price, self.config.tick_size, self.current_step
        )
        for side, price, qty in bg_orders:
            self.lob.add_order(-1, side, price, qty, self.current_step)

        # Match orders
        pnl, traded, traded_with_probe = self._process_trades()

        # Compute rewards
        rewards = self._compute_rewards(pnl)

        # Record metrics
        mid = self.lob.get_mid_price()
        spread = self.lob.get_spread()
        if mid:
            self.price_history.append(mid)
        if spread:
            self.spread_history.append(spread)

        # Get observations
        observations = {i: self._get_observation(i) for i in range(self.n_agents)}

        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length

        # Infos - includes trade information for probe detection
        infos = {
            i: {
                'inventory': self.inventories[i],
                'cash': self.cash[i],
                'mid_price': mid,
                'spread': spread,
                'fundamental': self.ou_process.x,
                'traded': traded[i],  # Whether agent traded this step
                'traded_with_probe': traded_with_probe[i],  # Whether agent traded with governance probe
            }
            for i in range(self.n_agents)
        }
        # Also include global dicts for easier access
        infos['trades'] = traded
        infos['probe_trades'] = traded_with_probe

        return observations, rewards, terminated, truncated, infos

    def get_metrics(self) -> Dict[str, float]:
        """Get environment metrics for logging."""
        return {
            'avg_spread': np.mean(self.spread_history) if self.spread_history else 0,
            'spread_std': np.std(self.spread_history) if self.spread_history else 0,
            'price_volatility': np.std(self.price_history) if len(self.price_history) > 1 else 0,
            'n_trades': len(self.trade_history),
            'avg_inventory': np.mean(np.abs(self.inventories)),
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            mid = self.lob.get_mid_price() or 0
            spread = self.lob.get_spread() or 0
            print(f"Step {self.current_step}: Mid={mid:.2f}, Spread={spread:.4f}, "
                  f"Inventories={self.inventories}")


# =============================================================================
# Factory function for easy creation
# =============================================================================

def make_lob_env(n_agents: int = 3, config: Optional[EnvironmentConfig] = None,
                 **kwargs) -> LOBEnv:
    """Create LOB environment with specified configuration."""
    return LOBEnv(config=config, n_agents=n_agents, **kwargs)


if __name__ == "__main__":
    # Test the environment
    print("Testing LOBEnv...")

    env = LOBEnv(n_agents=3)
    obs, info = env.reset(seed=42)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observations shape: {obs[0].shape}")

    # Run a few steps
    for step in range(10):
        actions = {i: env.action_space.sample() for i in range(3)}
        obs, rewards, terminated, truncated, infos = env.step(actions)

        print(f"Step {step}: Rewards={[rewards[i] for i in range(3)]}, "
              f"Spread={infos[0].get('spread', 'N/A')}")

        if terminated or truncated:
            break

    print("\nEnvironment metrics:", env.get_metrics())
    print("Test passed!")
