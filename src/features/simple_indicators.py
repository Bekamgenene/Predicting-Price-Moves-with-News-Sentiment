import pandas as pd
import numpy as np
from typing import List


def add_moving_averages(df: pd.DataFrame, 
                        periods: List[int] = [5, 10, 20, 50, 200],
                        price_col: str = 'Close') -> pd.DataFrame:
    result = df.copy()
    
    for period in periods:
        result[f'sma_{period}'] = result[price_col].rolling(window=period).mean()
        result[f'ema_{period}'] = result[price_col].ewm(span=period, adjust=False).mean()
    
    return result


def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'Close') -> pd.DataFrame:
    result = df.copy()
    delta = result[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return result


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, price_col: str = 'Close') -> pd.DataFrame:
    result = df.copy()
    result['bb_middle'] = result[price_col].rolling(window=period).mean()
    rolling_std = result[price_col].rolling(window=period).std()
    result['bb_upper'] = result['bb_middle'] + (rolling_std * std_dev)
    result['bb_lower'] = result['bb_middle'] - (rolling_std * std_dev)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    return result


def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_col: str = 'Close') -> pd.DataFrame:
    result = df.copy()
    ema_fast = result[price_col].ewm(span=fast_period, adjust=False).mean()
    ema_slow = result[price_col].ewm(span=slow_period, adjust=False).mean()
    result['macd_line'] = ema_fast - ema_slow
    result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
    result['macd_histogram'] = result['macd_line'] - result['macd_signal']
    return result


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    result = df.copy()
    low_min = result['Low'].rolling(window=k_period).min()
    high_max = result['High'].rolling(window=k_period).max()
    result['stoch_k'] = 100 * ((result['Close'] - low_min) / (high_max - low_min))
    result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
    return result


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = df.copy()
    result['tr1'] = abs(result['High'] - result['Low'])
    result['tr2'] = abs(result['High'] - result['Close'].shift())
    result['tr3'] = abs(result['Low'] - result['Close'].shift())
    result[f'atr_{period}'] = result[['tr1', 'tr2', 'tr3']].max(axis=1).rolling(window=period).mean()
    result = result.drop(['tr1', 'tr2', 'tr3'], axis=1)
    return result


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    
    if 'Volume' not in result.columns:
        print("Warning: Volume column not found. Volume indicators will not be calculated.")
        return result

    # On-Balance Volume (OBV) - vectorized
    delta = result['Close'].diff().fillna(0)
    direction = np.sign(delta)
    obv = np.zeros(len(result))
    for i in range(1, len(result)):
        obv[i] = obv[i-1] + direction[i] * result.loc[i, 'Volume']
    result['obv'] = obv

    # Volume-Weighted Average Price (VWAP)
    result['vwap'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()

    # Price-Volume Trend (PVT)
    result['pvt'] = ((result['Close'].diff() / result['Close'].shift()) * result['Volume']).cumsum()

    return result


def add_all_simple_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    try:
        print("Adding simple technical indicators...")
        print("  Adding moving averages...")
        result = add_moving_averages(result)
        print("  Adding RSI...")
        result = add_rsi(result)
        print("  Adding Bollinger Bands...")
        result = add_bollinger_bands(result)
        print("  Adding MACD...")
        result = add_macd(result)
        print("  Adding Stochastic Oscillator...")
        result = add_stochastic(result)
        print("  Adding ATR...")
        result = add_atr(result)
        print("  Adding volume indicators...")
        result = add_volume_indicators(result)
        print("All simple indicators added successfully!")
        return result
    except Exception as e:
        print(f"Error adding simple indicators: {e}")
        return df


def generate_simple_signals(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    try:
        result['buy_signal'] = 0
        result['sell_signal'] = 0

        if 'rsi_14' in result.columns:
            result.loc[(result['rsi_14'] > 30) & (result['rsi_14'].shift(1) <= 30), 'buy_signal'] = 1
            result.loc[(result['rsi_14'] < 70) & (result['rsi_14'].shift(1) >= 70), 'sell_signal'] = 1

        if 'macd_line' in result.columns and 'macd_signal' in result.columns:
            result.loc[(result['macd_line'] > result['macd_signal']) & 
                       (result['macd_line'].shift(1) <= result['macd_signal'].shift(1)), 'buy_signal'] = 1
            result.loc[(result['macd_line'] < result['macd_signal']) & 
                       (result['macd_line'].shift(1) >= result['macd_signal'].shift(1)), 'sell_signal'] = 1

        if 'bb_lower' in result.columns and 'bb_upper' in result.columns:
            result.loc[(result['Close'] > result['bb_lower']) & 
                       (result['Close'].shift(1) <= result['bb_lower'].shift(1)), 'buy_signal'] = 1
            result.loc[(result['Close'] < result['bb_upper']) & 
                       (result['Close'].shift(1) >= result['bb_upper'].shift(1)), 'sell_signal'] = 1

        if 'sma_20' in result.columns and 'sma_50' in result.columns:
            result.loc[(result['sma_20'] > result['sma_50']) & 
                       (result['sma_20'].shift(1) <= result['sma_50'].shift(1)), 'buy_signal'] = 1
            result.loc[(result['sma_20'] < result['sma_50']) & 
                       (result['sma_20'].shift(1) >= result['sma_50'].shift(1)), 'sell_signal'] = 1

        result['signal_strength'] = result['buy_signal'] - result['sell_signal']
        print("Trading signals generated successfully")
        return result
    except Exception as e:
        print(f"Error generating signals: {e}")
        return result
