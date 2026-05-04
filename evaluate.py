import sys
from stable_baselines3 import PPO
from env import StockTradingEnv
from functions import getStockDataVec, formatPrice

def evaluate():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py [stock] [model_name]")
        return

    stock_name, model_name = sys.argv[1], sys.argv[2]
    
    # In practice, window_size should match the trained model's window_size
    # For this script, we'll try to infer it or keep it consistent (e.g., 10)
    window_size = 10 
    
    data = getStockDataVec(stock_name)
    env = StockTradingEnv(data, window_size)
    
    model = PPO.load("models/" + model_name, env=env)
    
    obs, _ = env.reset()
    done = False
    
    print(f"Evaluating model {model_name} on {stock_name}...")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Optional: Print actions as they happen
        # if action == 1: print(f"Buy at {data[env.t-1]}")
        # elif action == 2: print(f"Sell at {data[env.t-1]}")

    print("--------------------------------")
    print(f"Total Profit: {formatPrice(info['total_profit'])}")
    print("--------------------------------")

if __name__ == "__main__":
    evaluate()
