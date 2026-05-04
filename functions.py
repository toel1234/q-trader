import numpy as np
import pandas as pd
import math

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    df = pd.read_csv("data/" + key + ".csv")
    return df['Close'].values.tolist()

# returns the sigmoid
def sigmoid(x):
    try:
        if x < -709: # Avoid overflow
             return 0.0
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])
