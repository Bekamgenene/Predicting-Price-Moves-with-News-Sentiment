import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate daily returns for a stock.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with added return columns
    """
    result = df.copy()
    
    # Daily returns
    result['daily_return'] = result[price_col].pct_change()
    
    # Cumulative returns
    result['cumulative_return'] = (1 + result['daily_return']).cumprod() - 1
    
    # Log returns
    result['log_return'] = np.log(result[price_col] / result[price_col].shift(1))
    
    return result

def calculate_volatility(df: pd.DataFrame, window: int = 20, returns_col: str = 'daily_return') -> pd.DataFrame:
    """
    Calculate rolling volatility for a stock.
    
    Args:
        df: DataFrame with return data
        window: Window size for rolling volatility
        returns_col: Column name for returns
        
    Returns:
        DataFrame with added volatility columns
    """
    result = df.copy()
    
    # Annualized volatility (standard deviation of returns * sqrt(252))
    result[f'volatility_{window}d'] = result[returns_col].rolling(window=window).std() * np.sqrt(252)
    
    return result

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02, 
                        returns_col: str = 'daily_return', window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        df: DataFrame with return data
        risk_free_rate: Annual risk-free rate
        returns_col: Column name for returns
        window: Window size for rolling calculation
        
    Returns:
        DataFrame with added Sharpe ratio column
    """
    result = df.copy()
    
    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Rolling Sharpe ratio
    excess_return = result[returns_col] - daily_rf
    result[f'sharpe_ratio_{window}d'] = (
        excess_return.rolling(window=window).mean() / 
        excess_return.rolling(window=window).std() * 
        np.sqrt(252)
    )
    
    return result

def calculate_drawdowns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate drawdowns for a stock.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with added drawdown columns
    """
    result = df.copy()
    
    # Calculate rolling maximum
    result['rolling_max'] = result[price_col].cummax()
    
    # Calculate drawdown
    result['drawdown'] = result[price_col] / result['rolling_max'] - 1
    
    return result

def calculate_beta(stock_df: pd.DataFrame, market_df: pd.DataFrame, 
                stock_returns_col: str = 'daily_return', 
                market_returns_col: str = 'daily_return',
                window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling beta for a stock relative to the market.
    
    Args:
        stock_df: DataFrame with stock return data
        market_df: DataFrame with market return data
        stock_returns_col: Column name for stock returns
        market_returns_col: Column name for market returns
        window: Window size for rolling calculation
        
    Returns:
        DataFrame with added beta column
    """
    result = stock_df.copy()
    
    # Align dates between stock and market data
    if isinstance(market_df.index, pd.DatetimeIndex) and isinstance(result.index, pd.DatetimeIndex):
        market_returns = market_df[market_returns_col].reindex(result.index)
    else:
        # If indices are not DatetimeIndex, do a simple merge
        market_returns = market_df[market_returns_col]
    
    # Calculate rolling covariance and market variance
    rolling_cov = result[stock_returns_col].rolling(window=window).cov(market_returns)
    rolling_var = market_returns.rolling(window=window).var()
    
    # Calculate beta
    result[f'beta_{window}d'] = rolling_cov / rolling_var
    
    return result

def calculate_alpha(stock_df: pd.DataFrame, market_df: pd.DataFrame, 
                  risk_free_rate: float = 0.02,
                  stock_returns_col: str = 'daily_return',
                  market_returns_col: str = 'daily_return',
                  beta_col: str = 'beta_252d',
                  window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling alpha for a stock.
    
    Args:
        stock_df: DataFrame with stock return data
        market_df: DataFrame with market return data
        risk_free_rate: Annual risk-free rate
        stock_returns_col: Column name for stock returns
        market_returns_col: Column name for market returns
        beta_col: Column name for beta
        window: Window size for rolling calculation
        
    Returns:
        DataFrame with added alpha column
    """
    result = stock_df.copy()
    
    # Check if beta column exists
    if beta_col not in result.columns:
        result = calculate_beta(result, market_df, stock_returns_col, market_returns_col, window)
    
    # Align dates between stock and market data
    if isinstance(market_df.index, pd.DatetimeIndex) and isinstance(result.index, pd.DatetimeIndex):
        market_returns = market_df[market_returns_col].reindex(result.index)
    else:
        market_returns = market_df[market_returns_col]
    
    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate alpha using CAPM: R_stock = Alpha + Beta * R_market + Risk-free rate
    result[f'alpha_{window}d'] = (
        result[stock_returns_col].rolling(window=window).mean() - 
        (daily_rf + result[beta_col] * (market_returns.rolling(window=window).mean() - daily_rf))
    ) * 252  # Annualize
    
    return result

def calculate_all_metrics(stock_df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate all financial metrics for a stock.
    
    Args:
        stock_df: DataFrame with stock price data
        market_df: DataFrame with market price data (for beta/alpha calculations)
        
    Returns:
        DataFrame with all financial metrics
    """
    result = stock_df.copy()
    
    # Calculate returns
    result = calculate_returns(result)
    
    # Calculate volatility
    result = calculate_volatility(result)
    
    # Calculate Sharpe ratio
    result = calculate_sharpe_ratio(result)
    
    # Calculate drawdowns
    result = calculate_drawdowns(result)
    
    # Calculate beta and alpha if market data is provided
    if market_df is not None:
        market_df = calculate_returns(market_df)
        result = calculate_beta(result, market_df)
        result = calculate_alpha(result, market_df)
    
    return result
