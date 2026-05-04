import optuna
import sys
import os
import numpy as np
from stable_baselines3 import PPO
from env import StockTradingEnv
from functions import getStockDataVec
from models import TransformerExtractor

def objective(trial):
    # Hyperparameters to tune
    window_size = trial.suggest_int("window_size", 5, 30)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    features_dim = trial.suggest_categorical("features_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    
    # Load data
    stock_name = "^GSPC" # Default for tuning
    data = getStockDataVec(stock_name)
    
    # Split data for simple validation (e.g., 80% train, 20% val)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]
    
    # Setup environments
    train_env = StockTradingEnv(train_data, window_size)
    val_env = StockTradingEnv(val_data, window_size)
    
    # Setup model with custom Transformer features extractor
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ),
    )
    
    model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, learning_rate=learning_rate, verbose=0)
    
    # Train for a moderate number of steps for tuning
    model.learn(total_timesteps=10000)
    
    # Evaluate on validation data
    obs, _ = val_env.reset()
    done = False
    total_profit = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = val_env.step(action)
    
    total_profit = info["total_profit"]
    
    # Optuna aims to maximize this value
    return total_profit

def tune():
    study_name = "qtrader_transformer_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True,
        direction="maximize"
    )
    
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=20)
    
    print("Optimization complete.")
    print("Best hyperparameters:", study.best_params)
    print("Best profit:", study.best_value)

if __name__ == "__main__":
    tune()
