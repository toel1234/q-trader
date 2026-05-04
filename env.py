import gymnasium as gym
from gymnasium import spaces
import numpy as np
from functions import getState, formatPrice

class StockTradingEnv(gym.Env):
    """
    Custom Stock Trading Environment that follows gymnasium interface.
    Actions: 0: Sit, 1: Buy, 2: Sell
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.l = len(data) - 1
        
        # Action space: 0: Sit, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: sequence of normalized price changes
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, window_size), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.inventory = []
        self.total_profit = 0
        self.done = False
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        state = getState(self.data, self.t, self.window_size + 1)
        return state.astype(np.float32), {}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        action = int(action)
        
        reward = 0
        current_price = self.data[self.t]
        self.action_counts[action] += 1
        
        if action == 1: # Buy
            self.inventory.append(current_price)
            # reward = -0.01 # small cost for buying?
        elif action == 2 and len(self.inventory) > 0: # Sell
            bought_price = self.inventory.pop(0)
            profit = current_price - bought_price
            self.total_profit += profit
            
            # Significant reward for profit, penalty for loss
            if profit > 0:
                reward = 1.0 + (profit / bought_price) * 10 
            else:
                reward = -1.0
        
        # Small penalty for holding inventory too long or sitting idle when price moves?
        # For now, keep it simple.
            
        self.t += 1
        self.done = True if self.t == self.l else False
        
        if self.done:
            print(f"Episode Done. Actions: {self.action_counts}, Profit: {formatPrice(self.total_profit)}")

        next_state = getState(self.data, self.t, self.window_size + 1)
        
        info = {"total_profit": self.total_profit, "inventory_count": len(self.inventory)}
        
        return next_state.astype(np.float32), reward, self.done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.t}, Profit: {formatPrice(self.total_profit)}, Inventory: {len(self.inventory)}")

    def close(self):
        pass
