"""
Technical indicators for stock price analysis
"""

import pandas as pd
import numpy as np

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=window).mean()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    std = prices.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return pd.DataFrame({
        'SMA': sma,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band,
        'Band_Width': upper_band - lower_band,
        'Percent_B': (prices - lower_band) / (upper_band - lower_band)
    })
