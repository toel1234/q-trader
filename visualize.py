import sys
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import StockTradingEnv
from functions import getStockDataVec, formatPrice

def visualize():
    if len(sys.argv) != 3:
        print("Usage: python visualize.py [stock] [model_name]")
        return

    stock_name, model_name = sys.argv[1], sys.argv[2]
    window_size = 10 
    
    data = getStockDataVec(stock_name)
    env = StockTradingEnv(data, window_size)
    
    model_path = "models/" + model_name
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
         model_path = model_name # try absolute or direct path

    model = PPO.load(model_path, env=env)
    
    obs, _ = env.reset()
    done = False
    
    buy_signals = []
    sell_signals = []
    prices = data
    
    print(f"Visualizing model {model_name} on {stock_name}...")
    
    t = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Record actions for plotting
        if action == 1: # Buy
            buy_signals.append((t, prices[t]))
        elif action == 2 and len(env.inventory) > 0: # Sell
            sell_signals.append((t, prices[t]))
            
        obs, reward, done, truncated, info = env.step(action)
        t += 1

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Stock Price', color='blue', alpha=0.6)
    
    if buy_signals:
        buy_indices, buy_prices = zip(*buy_signals)
        plt.scatter(buy_indices, buy_prices, marker='^', color='green', label='Buy', s=100)
        
    if sell_signals:
        sell_indices, sell_prices = zip(*sell_signals)
        plt.scatter(sell_indices, sell_prices, marker='v', color='red', label='Sell', s=100)
    
    plt.title(f"Trading Results: {stock_name} | Total Profit: {formatPrice(info['total_profit'])}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists("images"):
        os.makedirs("images")
        
    save_path = f"images/{stock_name}_result.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    print(f"Total Profit: {formatPrice(info['total_profit'])}")

if __name__ == "__main__":
    visualize()
