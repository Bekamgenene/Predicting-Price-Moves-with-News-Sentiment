import pandas as pd
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
from ta import momentum, trend, volatility, volume

def add_ta_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators from the TA library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with momentum indicators added
    """
    result = df.copy()
    
    # RSI (Relative Strength Index)
    result['ta_rsi'] = momentum.RSIIndicator(
        close=result['Close'], 
        window=14, 
        fillna=True
    ).rsi()
    
    # Stochastic Oscillator
    stoch = momentum.StochasticOscillator(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        window=14,
        smooth_window=3,
        fillna=True
    )
    result['ta_stoch_k'] = stoch.stoch()
    result['ta_stoch_d'] = stoch.stoch_signal()
    
    # TSI (True Strength Index)
    result['ta_tsi'] = momentum.TSIIndicator(
        close=result['Close'],
        window_slow=25,
        window_fast=13,
        fillna=True
    ).tsi()
    
    # Ultimate Oscillator
    result['ta_uo'] = momentum.UltimateOscillator(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        window1=7,
        window2=14,
        window3=28,
        weight1=4.0,
        weight2=2.0,
        weight3=1.0,
        fillna=True
    ).ultimate_oscillator()
    
    # Williams %R
    result['ta_williams_r'] = momentum.WilliamsRIndicator(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        lbp=14,
        fillna=True
    ).williams_r()
    
    return result

def add_ta_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators from the TA library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with trend indicators added
    """
    result = df.copy()
    
    # MACD (Moving Average Convergence Divergence)
    macd = trend.MACD(
        close=result['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9,
        fillna=True
    )
    result['ta_macd'] = macd.macd()
    result['ta_macd_signal'] = macd.macd_signal()
    result['ta_macd_diff'] = macd.macd_diff()
    
    # SMA (Simple Moving Average)
    for period in [5, 10, 20, 50, 200]:
        result[f'ta_sma_{period}'] = trend.SMAIndicator(
            close=result['Close'],
            window=period,
            fillna=True
        ).sma_indicator()
    
    # EMA (Exponential Moving Average)
    for period in [5, 10, 20, 50, 200]:
        result[f'ta_ema_{period}'] = trend.EMAIndicator(
            close=result['Close'],
            window=period,
            fillna=True
        ).ema_indicator()
    
    # ADX (Average Directional Movement Index)
    adx = trend.ADXIndicator(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        window=14,
        fillna=True
    )
    result['ta_adx'] = adx.adx()
    result['ta_adx_pos'] = adx.adx_pos()
    result['ta_adx_neg'] = adx.adx_neg()
    
    # Ichimoku Cloud
    ichimoku = trend.IchimokuIndicator(
        high=result['High'],
        low=result['Low'],
        window1=9,
        window2=26,
        window3=52,
        visual=False,
        fillna=True
    )
    result['ta_ichimoku_a'] = ichimoku.ichimoku_a()
    result['ta_ichimoku_b'] = ichimoku.ichimoku_b()
    result['ta_ichimoku_base'] = ichimoku.ichimoku_base_line()
    result['ta_ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    
    return result

def add_ta_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators from the TA library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with volatility indicators added
    """
    result = df.copy()
    
    # Bollinger Bands
    bollinger = volatility.BollingerBands(
        close=result['Close'],
        window=20,
        window_dev=2,
        fillna=True
    )
    result['ta_bollinger_mavg'] = bollinger.bollinger_mavg()
    result['ta_bollinger_hband'] = bollinger.bollinger_hband()
    result['ta_bollinger_lband'] = bollinger.bollinger_lband()
    result['ta_bollinger_width'] = bollinger.bollinger_wband()
    
    # ATR (Average True Range)
    result['ta_atr'] = volatility.AverageTrueRange(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        window=14,
        fillna=True
    ).average_true_range()
    
    # Keltner Channel
    keltner = volatility.KeltnerChannel(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        window=20,
        window_atr=10,
        fillna=True,
        original_version=False
    )
    result['ta_keltner_high'] = keltner.keltner_channel_hband()
    result['ta_keltner_low'] = keltner.keltner_channel_lband()
    result['ta_keltner_mid'] = keltner.keltner_channel_mband()
    result['ta_keltner_width'] = keltner.keltner_channel_wband()
    
    # Donchian Channel
    donchian = volatility.DonchianChannel(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        window=20,
        fillna=True
    )
    result['ta_donchian_high'] = donchian.donchian_channel_hband()
    result['ta_donchian_low'] = donchian.donchian_channel_lband()
    result['ta_donchian_mid'] = donchian.donchian_channel_mband()
    result['ta_donchian_width'] = donchian.donchian_channel_wband()
    
    return result

def add_ta_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume indicators from the TA library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with volume indicators added
    """
    result = df.copy()
    
    # Ensure Volume column exists
    if 'Volume' not in result.columns:
        print("Warning: Volume column not found. Volume indicators will not be calculated.")
        return result
    
    # On-Balance Volume (OBV)
    result['ta_obv'] = volume.OnBalanceVolumeIndicator(
        close=result['Close'],
        volume=result['Volume'],
        fillna=True
    ).on_balance_volume()
    
    # Volume Weighted Average Price (VWAP)
    # Note: VWAP typically requires intraday data, so it's usually calculated per day
    # For daily data, we can calculate a rolling VWAP
    result['ta_vwap'] = volume.VolumeWeightedAveragePrice(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        volume=result['Volume'],
        window=14,
        fillna=True
    ).volume_weighted_average_price()
    
    # Chaikin Money Flow
    result['ta_cmf'] = volume.ChaikinMoneyFlowIndicator(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        volume=result['Volume'],
        window=20,
        fillna=True
    ).chaikin_money_flow()
    
    # Force Index
    result['ta_fi'] = volume.ForceIndexIndicator(
        close=result['Close'],
        volume=result['Volume'],
        window=13,
        fillna=True
    ).force_index()
    
    # Money Flow Index
    result['ta_mfi'] = volume.MFIIndicator(
        high=result['High'],
        low=result['Low'],
        close=result['Close'],
        volume=result['Volume'],
        window=14,
        fillna=True
    ).money_flow_index()
    
    # Negative Volume Index
    result['ta_nvi'] = volume.NegativeVolumeIndexIndicator(
        close=result['Close'],
        volume=result['Volume'],
        fillna=True
    ).negative_volume_index()
    
    # Volume Price Trend
    result['ta_vpt'] = volume.VolumePriceTrendIndicator(
        close=result['Close'],
        volume=result['Volume'],
        fillna=True
    ).volume_price_trend()
    
    return result

def add_all_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators from the TA library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all TA indicators
    """
    result = df.copy()
    
    try:
        print("Adding TA library indicators...")
        
        print("  Adding momentum indicators...")
        result = add_ta_momentum_indicators(result)
        
        print("  Adding trend indicators...")
        result = add_ta_trend_indicators(result)
        
        print("  Adding volatility indicators...")
        result = add_ta_volatility_indicators(result)
        
        print("  Adding volume indicators...")
        result = add_ta_volume_indicators(result)
        
        print("TA indicators added successfully!")
        return result
    except Exception as e:
        print(f"Error adding TA indicators: {e}")
        # If there's an error, return the original DataFrame
        return df

def get_ta_buy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy signals based on TA indicators.
    
    Args:
        df: DataFrame with TA indicators
        
    Returns:
        DataFrame with buy signals
    """
    result = df.copy()
    
    # Initialize signals column
    result['ta_buy_signal'] = 0
    
    # RSI oversold (RSI < 30)
    if 'ta_rsi' in result.columns:
        result.loc[result['ta_rsi'] < 30, 'ta_buy_signal'] = 1
    
    # MACD line crosses above signal line
    if 'ta_macd_diff' in result.columns:
        result.loc[(result['ta_macd_diff'] > 0) & (result['ta_macd_diff'].shift(1) <= 0), 'ta_buy_signal'] = 1
    
    # Price crosses above SMA 50
    if 'ta_sma_50' in result.columns:
        result.loc[(result['Close'] > result['ta_sma_50']) & (result['Close'].shift(1) <= result['ta_sma_50'].shift(1)), 'ta_buy_signal'] = 1
    
    # Bollinger Bands: Price touches lower band
    if 'ta_bollinger_lband' in result.columns:
        result.loc[result['Close'] <= result['ta_bollinger_lband'], 'ta_buy_signal'] = 1
    
    # Stochastic crossover in oversold region
    if 'ta_stoch_k' in result.columns and 'ta_stoch_d' in result.columns:
        result.loc[(result['ta_stoch_k'] < 20) & (result['ta_stoch_d'] < 20) & 
                  (result['ta_stoch_k'] > result['ta_stoch_d']) & 
                  (result['ta_stoch_k'].shift(1) <= result['ta_stoch_d'].shift(1)), 'ta_buy_signal'] = 1
    
    return result

def get_ta_sell_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sell signals based on TA indicators.
    
    Args:
        df: DataFrame with TA indicators
        
    Returns:
        DataFrame with sell signals
    """
    result = df.copy()
    
    # Initialize signals column
    result['ta_sell_signal'] = 0
    
    # RSI overbought (RSI > 70)
    if 'ta_rsi' in result.columns:
        result.loc[result['ta_rsi'] > 70, 'ta_sell_signal'] = 1
    
    # MACD line crosses below signal line
    if 'ta_macd_diff' in result.columns:
        result.loc[(result['ta_macd_diff'] < 0) & (result['ta_macd_diff'].shift(1) >= 0), 'ta_sell_signal'] = 1
    
    # Price crosses below SMA 50
    if 'ta_sma_50' in result.columns:
        result.loc[(result['Close'] < result['ta_sma_50']) & (result['Close'].shift(1) >= result['ta_sma_50'].shift(1)), 'ta_sell_signal'] = 1
    
    # Bollinger Bands: Price touches upper band
    if 'ta_bollinger_hband' in result.columns:
        result.loc[result['Close'] >= result['ta_bollinger_hband'], 'ta_sell_signal'] = 1
    
    # Stochastic crossover in overbought region
    if 'ta_stoch_k' in result.columns and 'ta_stoch_d' in result.columns:
        result.loc[(result['ta_stoch_k'] > 80) & (result['ta_stoch_d'] > 80) & 
                  (result['ta_stoch_k'] < result['ta_stoch_d']) & 
                  (result['ta_stoch_k'].shift(1) >= result['ta_stoch_d'].shift(1)), 'ta_sell_signal'] = 1
    
    return result

def get_ta_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy and sell signals based on TA indicators.
    
    Args:
        df: DataFrame with TA indicators
        
    Returns:
        DataFrame with trading signals
    """
    result = df.copy()
    
    # Generate buy signals
    result = get_ta_buy_signals(result)
    
    # Generate sell signals
    result = get_ta_sell_signals(result)
    
    return result
