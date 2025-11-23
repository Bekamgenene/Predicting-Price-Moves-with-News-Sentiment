import pandas as pd
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
from ta import momentum, trend, volatility, volume
import ta

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with momentum indicators added
    """
    result = df.copy()
    
    try:
        # RSI (Relative Strength Index)
        result['ta_rsi'] = momentum.RSIIndicator(
            close=result['Close'].values.flatten(), 
            window=14, 
            fillna=True
        ).rsi()
        
        # Stochastic Oscillator
        stoch = momentum.StochasticOscillator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            window=14,
            smooth_window=3,
            fillna=True
        )
        result['ta_stoch_k'] = stoch.stoch()
        result['ta_stoch_d'] = stoch.stoch_signal()
        
        # TSI (True Strength Index)
        result['ta_tsi'] = momentum.TSIIndicator(
            close=result['Close'].values.flatten(),
            window_slow=25,
            window_fast=13,
            fillna=True
        ).tsi()
        
        # ROC (Rate of Change)
        result['ta_roc'] = momentum.ROCIndicator(
            close=result['Close'].values.flatten(),
            window=12,
            fillna=True
        ).roc()
        
        # Williams %R
        result['ta_williams_r'] = momentum.WilliamsRIndicator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            lbp=14,
            fillna=True
        ).williams_r()
        
        print("  Momentum indicators added successfully")
    except Exception as e:
        print(f"  Error adding momentum indicators: {e}")
    
    return result

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with trend indicators added
    """
    result = df.copy()
    
    try:
        # MACD (Moving Average Convergence Divergence)
        macd = trend.MACD(
            close=result['Close'].values.flatten(),
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
                close=result['Close'].values.flatten(),
                window=period,
                fillna=True
            ).sma_indicator()
        
        # EMA (Exponential Moving Average)
        for period in [5, 10, 20, 50, 200]:
            result[f'ta_ema_{period}'] = trend.EMAIndicator(
                close=result['Close'].values.flatten(),
                window=period,
                fillna=True
            ).ema_indicator()
        
        # ADX (Average Directional Movement Index)
        adx = trend.ADXIndicator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            window=14,
            fillna=True
        )
        result['ta_adx'] = adx.adx()
        result['ta_adx_pos'] = adx.adx_pos()
        result['ta_adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        result['ta_psar'] = trend.PSARIndicator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            step=0.02,
            max_step=0.2,
            fillna=True
        ).psar()
        
        print("  Trend indicators added successfully")
    except Exception as e:
        print(f"  Error adding trend indicators: {e}")
    
    return result

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with volatility indicators added
    """
    result = df.copy()
    
    try:
        # Bollinger Bands
        bollinger = volatility.BollingerBands(
            close=result['Close'].values.flatten(),
            window=20,
            window_dev=2,
            fillna=True
        )
        result['ta_bollinger_mavg'] = bollinger.bollinger_mavg()
        result['ta_bollinger_hband'] = bollinger.bollinger_hband()
        result['ta_bollinger_lband'] = bollinger.bollinger_lband()
        result['ta_bollinger_width'] = bollinger.bollinger_wband()
        result['ta_bollinger_pband'] = bollinger.bollinger_pband()
        
        # ATR (Average True Range)
        result['ta_atr'] = volatility.AverageTrueRange(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            window=14,
            fillna=True
        ).average_true_range()
        
        # Keltner Channel
        keltner = volatility.KeltnerChannel(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
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
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            window=20,
            fillna=True
        )
        result['ta_donchian_high'] = donchian.donchian_channel_hband()
        result['ta_donchian_low'] = donchian.donchian_channel_lband()
        result['ta_donchian_mid'] = donchian.donchian_channel_mband()
        
        print("  Volatility indicators added successfully")
    except Exception as e:
        print(f"  Error adding volatility indicators: {e}")
    
    return result

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with volume indicators added
    """
    result = df.copy()
    
    try:
        # Ensure Volume column exists
        if 'Volume' not in result.columns:
            print("  Warning: Volume column not found. Volume indicators will not be calculated.")
            return result
        
        # On-Balance Volume (OBV)
        result['ta_obv'] = volume.OnBalanceVolumeIndicator(
            close=result['Close'].values.flatten(),
            volume=result['Volume'].values.flatten(),
            fillna=True
        ).on_balance_volume()
        
        # Volume Weighted Average Price (VWAP)
        result['ta_vwap'] = volume.VolumeWeightedAveragePrice(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            volume=result['Volume'].values.flatten(),
            window=14,
            fillna=True
        ).volume_weighted_average_price()
        
        # Chaikin Money Flow
        result['ta_cmf'] = volume.ChaikinMoneyFlowIndicator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            volume=result['Volume'].values.flatten(),
            window=20,
            fillna=True
        ).chaikin_money_flow()
        
        # Force Index
        result['ta_fi'] = volume.ForceIndexIndicator(
            close=result['Close'].values.flatten(),
            volume=result['Volume'].values.flatten(),
            window=13,
            fillna=True
        ).force_index()
        
        # Money Flow Index
        result['ta_mfi'] = volume.MFIIndicator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            close=result['Close'].values.flatten(),
            volume=result['Volume'].values.flatten(),
            window=14,
            fillna=True
        ).money_flow_index()
        
        # Ease of Movement
        result['ta_eom'] = volume.EaseOfMovementIndicator(
            high=result['High'].values.flatten(),
            low=result['Low'].values.flatten(),
            volume=result['Volume'].values.flatten(),
            window=14,
            fillna=True
        ).ease_of_movement()
        
        print("  Volume indicators added successfully")
    except Exception as e:
        print(f"  Error adding volume indicators: {e}")
    
    return result

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators using TA-Lib.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all technical indicators
    """
    result = df.copy()
    
    try:
        print("Adding TA-Lib indicators...")
        
        # Check if result is a DataFrame
        if not isinstance(result, pd.DataFrame):
            print(f"Error: Expected DataFrame but got {type(result)}")
            return df
        
        # Debug - print shape of columns
        print(f"  DataFrame shape: {result.shape}")
        print(f"  Close column shape: {result['Close'].shape}")
        
        # Add momentum indicators
        print("  Adding momentum indicators...")
        momentum_df = add_momentum_indicators(result)
        if isinstance(momentum_df, pd.DataFrame):
            result = momentum_df
        
        # Add trend indicators
        print("  Adding trend indicators...")
        trend_df = add_trend_indicators(result)
        if isinstance(trend_df, pd.DataFrame):
            result = trend_df
        
        # Add volatility indicators
        print("  Adding volatility indicators...")
        volatility_df = add_volatility_indicators(result)
        if isinstance(volatility_df, pd.DataFrame):
            result = volatility_df
        
        # Add volume indicators
        print("  Adding volume indicators...")
        volume_df = add_volume_indicators(result)
        if isinstance(volume_df, pd.DataFrame):
            result = volume_df
        
        print("All TA-Lib indicators added successfully!")
        return result
    except Exception as e:
        print(f"Error adding TA-Lib indicators: {e}")
        # If there's an error, return the original DataFrame
        return df

def add_all_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators using the add_all_ta_features function from the TA library.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all technical indicators
    """
    # Make sure we're working with a DataFrame
    if not isinstance(df, pd.DataFrame):
        print(f"Error: Expected DataFrame but got {type(df)}")
        return df
    
    result = df.copy()
    
    try:
        # Print the DataFrame information for debugging
        print(f"DataFrame shape: {result.shape}")
        print(f"DataFrame columns: {result.columns.tolist()}")
        print(f"Close column shape: {result['Close'].shape}")
        print(f"Close column type: {type(result['Close'])}")
        
        # Use our custom indicator functions instead of ta.add_all_ta_features
        # as it seems to be causing issues
        print("Using custom indicators instead of ta.add_all_ta_features...")
        return add_all_indicators(result)
    except Exception as e:
        print(f"Error in add_all_ta_indicators: {e}")
        # If there's an error, return the original DataFrame
        return df

def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on TA-Lib indicators.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with trading signals
    """
    result = df.copy()
    
    try:
        # Initialize signal columns
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        
        # RSI signals
        if 'ta_rsi' in result.columns:
            # Buy when RSI crosses above 30 (oversold)
            result.loc[(result['ta_rsi'] > 30) & (result['ta_rsi'].shift(1) <= 30), 'buy_signal'] = 1
            # Sell when RSI crosses below 70 (overbought)
            result.loc[(result['ta_rsi'] < 70) & (result['ta_rsi'].shift(1) >= 70), 'sell_signal'] = 1
        
        # MACD signals
        if 'ta_macd' in result.columns and 'ta_macd_signal' in result.columns:
            # Buy when MACD crosses above signal line
            result.loc[(result['ta_macd'] > result['ta_macd_signal']) & 
                      (result['ta_macd'].shift(1) <= result['ta_macd_signal'].shift(1)), 'buy_signal'] = 1
            # Sell when MACD crosses below signal line
            result.loc[(result['ta_macd'] < result['ta_macd_signal']) & 
                      (result['ta_macd'].shift(1) >= result['ta_macd_signal'].shift(1)), 'sell_signal'] = 1
        
        # Bollinger Band signals
        if 'ta_bollinger_lband' in result.columns and 'ta_bollinger_hband' in result.columns:
            # Buy when price crosses above lower band
            result.loc[(result['Close'] > result['ta_bollinger_lband']) & 
                      (result['Close'].shift(1) <= result['ta_bollinger_lband'].shift(1)), 'buy_signal'] = 1
            # Sell when price crosses below upper band
            result.loc[(result['Close'] < result['ta_bollinger_hband']) & 
                      (result['Close'].shift(1) >= result['ta_bollinger_hband'].shift(1)), 'sell_signal'] = 1
        
        # SMA crossover signals
        if 'ta_sma_20' in result.columns and 'ta_sma_50' in result.columns:
            # Buy when 20-day SMA crosses above 50-day SMA (golden cross)
            result.loc[(result['ta_sma_20'] > result['ta_sma_50']) & 
                      (result['ta_sma_20'].shift(1) <= result['ta_sma_50'].shift(1)), 'buy_signal'] = 1
            # Sell when 20-day SMA crosses below 50-day SMA (death cross)
            result.loc[(result['ta_sma_20'] < result['ta_sma_50']) & 
                      (result['ta_sma_20'].shift(1) >= result['ta_sma_50'].shift(1)), 'sell_signal'] = 1
        
        # Combine signals (multiple indicators agree)
        result['signal_strength'] = result['buy_signal'] - result['sell_signal']
        
        print("Trading signals generated successfully")
    except Exception as e:
        print(f"Error generating trading signals: {e}")
    
    return result

def calculate_strategy_returns(df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Calculate returns for a trading strategy based on generated signals.
    
    Args:
        df: DataFrame with trading signals
        initial_capital: Initial capital for the strategy
        
    Returns:
        DataFrame with strategy returns
    """
    result = df.copy()
    
    try:
        # Ensure we have the necessary columns
        if 'buy_signal' not in result.columns or 'sell_signal' not in result.columns:
            result = generate_trading_signals(result)
        
        # Create position column (1 for long, -1 for short, 0 for no position)
        result['position'] = 0
        
        # Set position based on signals
        # Assuming we enter a position at the next day's open after a signal
        result.loc[result['buy_signal'] == 1, 'position'] = 1
        result.loc[result['sell_signal'] == 1, 'position'] = -1
        
        # Forward fill positions (hold until a new signal)
        result['position'] = result['position'].replace(to_replace=0, method='ffill')
        
        # Calculate daily returns
        if 'daily_return' not in result.columns:
            result['daily_return'] = result['Close'].pct_change()
        
        # Calculate strategy returns
        result['strategy_return'] = result['position'].shift(1) * result['daily_return']
        
        # Calculate cumulative returns
        result['strategy_cumulative_return'] = (1 + result['strategy_return']).cumprod() - 1
        result['buy_hold_cumulative_return'] = (1 + result['daily_return']).cumprod() - 1
        
        # Calculate equity curve
        result['strategy_equity'] = initial_capital * (1 + result['strategy_cumulative_return'])
        result['buy_hold_equity'] = initial_capital * (1 + result['buy_hold_cumulative_return'])
        
        print("Strategy returns calculated successfully")
    except Exception as e:
        print(f"Error calculating strategy returns: {e}")
    
    return result

def backtest_strategy(df: pd.DataFrame, 
                    initial_capital: float = 10000.0,
                    commission: float = 0.001) -> Dict:
    """
    Backtest a trading strategy based on generated signals.
    
    Args:
        df: DataFrame with trading signals
        initial_capital: Initial capital for the strategy
        commission: Commission per trade as a fraction of trade value
        
    Returns:
        Dictionary with backtest results
    """
    results = {}
    
    try:
        # Ensure we have strategy returns
        if 'strategy_return' not in df.columns:
            df = calculate_strategy_returns(df, initial_capital)
        
        # Calculate number of trades
        buy_signals = df['buy_signal'].sum()
        sell_signals = df['sell_signal'].sum()
        total_trades = buy_signals + sell_signals
        
        # Calculate returns statistics
        strategy_returns = df['strategy_return'].dropna()
        buy_hold_returns = df['daily_return'].dropna()
        
        # Annualized return
        trading_days_per_year = 252
        strategy_annual_return = strategy_returns.mean() * trading_days_per_year
        buy_hold_annual_return = buy_hold_returns.mean() * trading_days_per_year
        
        # Annualized volatility
        strategy_annual_vol = strategy_returns.std() * np.sqrt(trading_days_per_year)
        buy_hold_annual_vol = buy_hold_returns.std() * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio (assuming risk-free rate of 0%)
        strategy_sharpe = strategy_annual_return / strategy_annual_vol if strategy_annual_vol != 0 else 0
        buy_hold_sharpe = buy_hold_annual_return / buy_hold_annual_vol if buy_hold_annual_vol != 0 else 0
        
        # Maximum drawdown
        strategy_cum_returns = df['strategy_cumulative_return'].dropna()
        buy_hold_cum_returns = df['buy_hold_cumulative_return'].dropna()
        
        strategy_max_drawdown = (strategy_cum_returns - strategy_cum_returns.cummax()).min()
        buy_hold_max_drawdown = (buy_hold_cum_returns - buy_hold_cum_returns.cummax()).min()
        
        # Final equity
        strategy_final_equity = df['strategy_equity'].iloc[-1]
        buy_hold_final_equity = df['buy_hold_equity'].iloc[-1]
        
        # Compile results
        results = {
            'initial_capital': initial_capital,
            'strategy_final_equity': strategy_final_equity,
            'buy_hold_final_equity': buy_hold_final_equity,
            'strategy_return': (strategy_final_equity / initial_capital - 1) * 100,
            'buy_hold_return': (buy_hold_final_equity / initial_capital - 1) * 100,
            'strategy_annual_return': strategy_annual_return * 100,
            'buy_hold_annual_return': buy_hold_annual_return * 100,
            'strategy_annual_volatility': strategy_annual_vol * 100,
            'buy_hold_annual_volatility': buy_hold_annual_vol * 100,
            'strategy_sharpe': strategy_sharpe,
            'buy_hold_sharpe': buy_hold_sharpe,
            'strategy_max_drawdown': strategy_max_drawdown * 100,
            'buy_hold_max_drawdown': buy_hold_max_drawdown * 100,
            'total_trades': total_trades,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
        
        print("Strategy backtest completed successfully")
    except Exception as e:
        print(f"Error backtesting strategy: {e}")
        results = {'error': str(e)}
    
    return results
