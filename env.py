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
        
        state = getState(self.data, self.t, self.window_size + 1)
        return state.astype(np.float32), {}

    def step(self, action):
        reward = 0
        current_price = self.data[self.t]
        
        if action == 1: # Buy
            self.inventory.append(current_price)
        elif action == 2 and len(self.inventory) > 0: # Sell
            bought_price = self.inventory.pop(0)
            reward = max(current_price - bought_price, 0)
            self.total_profit += current_price - bought_price
            
        self.t += 1
        self.done = True if self.t == self.l else False
        
        next_state = getState(self.data, self.t, self.window_size + 1)
        
        info = {"total_profit": self.total_profit, "inventory_count": len(self.inventory)}
        
        return next_state.astype(np.float32), reward, self.done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.t}, Profit: {formatPrice(self.total_profit)}, Inventory: {len(self.inventory)}")

    def close(self):
        pass
