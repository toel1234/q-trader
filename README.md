# Q-Trader 2026 (Modernized)

A State-of-the-Art 2026 implementation of Reinforcement Learning applied to stock trading. This project utilizes a **Transformer-based RL** (PPO/A2C) agent built with **Stable Baselines3** and a custom **Gymnasium** environment.

The model uses n-day windows of normalized price changes to determine the optimal action (Buy, Sell, or Sit) using a deep **Transformer Encoder** as a feature extractor to capture complex temporal dependencies and market trends.

## Key Features (2026 Upgrade)
- **Architecture:** PyTorch Transformer Encoder (Self-Attention) for sequence feature extraction.
- **RL Algorithms:** Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C) via Stable Baselines3.
- **Environment:** Standardized Gymnasium `StockTradingEnv` with refined reward signals.
- **Optimization:** Automated hyperparameter tuning via **Optuna**.
- **Modern Stack:** Python 3.12+, Pandas, and PyTorch for high-performance execution.

## Results (2026 Modernized)

After extensive training on the S&P 500 (^GSPC) with **Optuna-optimized hyperparameters**, the agent achieved significant performance gains:

![^GSPC 2011 Result](images/^GSPC_2011_result.png)
**S&P 500, 2011. Profit of $27,253.67.**

The architecture effectively identifies strategic entry and exit points, leveraging the Transformer's ability to focus on critical price shifts.

## Running the Code

### 1. Hyperparameter Tuning
Find the optimal model configuration for a specific stock:
```bash
python tune.py
```
This script uses **Optuna** to search for the best `window_size`, `learning_rate`, and Transformer depth. Results are persisted in a local SQLite database.

### 2. Training
Train the agent using the optimized settings:
```bash
python train.py [stock_symbol] [window_size] [episodes]
```
Example:
```bash
python train.py ^GSPC 8 500
```

### 3. Visualization
Generate a trading performance plot with Buy/Sell markers:
```bash
python visualize.py [stock_symbol] [model_name]
```
Example:
```bash
python visualize.py ^GSPC_2011 ppo_transformer_^GSPC
```

### 4. Evaluation
Run a simple deterministic evaluation to check total profit:
```bash
python evaluate.py [stock_symbol] [model_name]
```

## Data
Stock data should be placed in the `data/` directory as CSV files (standard Yahoo Finance format).

## Implementation Details
- **Observation Space:** n-day window of price change sigmoids (Shape: `1 x window_size`).
- **Action Space:** Discrete (0: Sit, 1: Buy, 2: Sell).
- **Reward Function:** Enhanced signal based on realized profit percentage with penalties for inefficiency.
