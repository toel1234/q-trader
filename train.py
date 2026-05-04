import sys
import os
from stable_baselines3 import PPO
from env import StockTradingEnv
from functions import getStockDataVec
from models import TransformerExtractor

def train():
    if len(sys.argv) != 4:
        print("Usage: python train.py [stock] [window] [episodes]")
        return

    stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    
    data = getStockDataVec(stock_name)
    env = StockTradingEnv(data, window_size)
    
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )
    
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    
    # Estimate total steps based on episodes and data length
    total_timesteps = episode_count * len(data)
    
    print(f"Starting training for {episode_count} episodes ({total_timesteps} steps)...")
    model.learn(total_timesteps=total_timesteps)
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model_path = f"models/ppo_transformer_{stock_name}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
