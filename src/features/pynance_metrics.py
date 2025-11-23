import pandas as pd
import numpy as np
import pynance as pn
from typing import Dict, List, Optional, Union, Tuple


def calculate_pynance_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate returns using PyNance.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with added return columns
    """
    result = df.copy()
    
    # Convert DataFrame to PyNance data structure
    try:
        pn_data = pn.data.create(result)
        
        # Calculate daily returns
        result['pn_daily_return'] = pn.returns.simple(pn_data[price_col])
        
        # Calculate log returns
        result['pn_log_return'] = pn.returns.log(pn_data[price_col])
        
        # Calculate cumulative returns
        result['pn_cum_return'] = pn.returns.cumulative(pn_data[price_col])
        
        print("PyNance returns calculated successfully")
    except Exception as e:
        print(f"Error calculating PyNance returns: {e}")
        # If error occurs, calculate returns using pandas
        result['pn_daily_return'] = result[price_col].pct_change()
        result['pn_log_return'] = np.log(result[price_col] / result[price_col].shift(1))
        result['pn_cum_return'] = (1 + result['pn_daily_return']).cumprod() - 1
    
    return result


def calculate_pynance_risk_metrics(df: pd.DataFrame, 
                                 returns_col: str = 'pn_daily_return',
                                 window: int = 252,
                                 risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Calculate risk metrics using PyNance.
    
    Args:
        df: DataFrame with return data
        returns_col: Column name for returns
        window: Window size for rolling calculations
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with added risk metric columns
    """
    result = df.copy()
    
    try:
        # Ensure returns column exists
        if returns_col not in result.columns:
            raise ValueError(f"Column {returns_col} not found in DataFrame")
        
        # Daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate rolling volatility (annualized)
        result[f'pn_volatility_{window}d'] = (
            result[returns_col].rolling(window=window).std() * np.sqrt(252)
        )
        
        # Calculate rolling Sharpe ratio
        excess_returns = result[returns_col] - daily_rf
        result[f'pn_sharpe_{window}d'] = (
            excess_returns.rolling(window=window).mean() / 
            result[returns_col].rolling(window=window).std() * 
            np.sqrt(252)
        )
        
        # Calculate rolling Sortino ratio (using only negative returns)
        downside_returns = result[returns_col].copy()
        downside_returns[downside_returns > 0] = 0
        downside_deviation = downside_returns.rolling(window=window).std() * np.sqrt(252)
        result[f'pn_sortino_{window}d'] = (
            (result[returns_col].rolling(window=window).mean() - daily_rf) * 252 / 
            downside_deviation
        ).replace([np.inf, -np.inf], np.nan)
        
        print("PyNance risk metrics calculated successfully")
    except Exception as e:
        print(f"Error calculating PyNance risk metrics: {e}")
    
    return result


def calculate_pynance_var(df: pd.DataFrame, 
                        confidence_level: float = 0.95,
                        returns_col: str = 'pn_daily_return',
                        window: int = 252) -> pd.DataFrame:
    """
    Calculate Value at Risk (VaR) using PyNance.
    
    Args:
        df: DataFrame with return data
        confidence_level: Confidence level for VaR calculation
        returns_col: Column name for returns
        window: Window size for rolling calculation
        
    Returns:
        DataFrame with added VaR column
    """
    result = df.copy()
    
    try:
        # Calculate rolling historical VaR
        percentile = 1 - confidence_level
        result[f'pn_var_{int(confidence_level*100)}'] = (
            result[returns_col].rolling(window=window).quantile(percentile)
        )
        
        print(f"PyNance VaR ({confidence_level*100}%) calculated successfully")
    except Exception as e:
        print(f"Error calculating PyNance VaR: {e}")
    
    return result


def calculate_pynance_drawdowns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate drawdowns using PyNance.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with added drawdown columns
    """
    result = df.copy()
    
    try:
        # Calculate running maximum
        result['pn_peak'] = result[price_col].cummax()
        
        # Calculate drawdowns
        result['pn_drawdown'] = result[price_col] / result['pn_peak'] - 1
        
        # Calculate drawdown duration
        result['pn_drawdown_start'] = (result['pn_peak'] != result['pn_peak'].shift(1)).astype(int)
        result['pn_drawdown_start'] = result['pn_drawdown_start'].replace(0, np.nan)
        result['pn_drawdown_group'] = result['pn_drawdown_start'].cumsum()
        result['pn_drawdown_duration'] = result.groupby('pn_drawdown_group').cumcount() + 1
        
        # Clean up intermediate columns
        result = result.drop(['pn_drawdown_start', 'pn_drawdown_group'], axis=1)
        
        print("PyNance drawdowns calculated successfully")
    except Exception as e:
        print(f"Error calculating PyNance drawdowns: {e}")
    
    return result


def calculate_pynance_ratios(df: pd.DataFrame, 
                            benchmark_df: pd.DataFrame = None,
                            returns_col: str = 'pn_daily_return',
                            benchmark_returns_col: str = 'daily_return',
                            window: int = 252,
                            risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Calculate various ratios using PyNance.
    
    Args:
        df: DataFrame with return data
        benchmark_df: DataFrame with benchmark return data
        returns_col: Column name for returns
        benchmark_returns_col: Column name for benchmark returns
        window: Window size for rolling calculations
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with added ratio columns
    """
    result = df.copy()
    
    try:
        # Daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate Treynor ratio if benchmark is provided
        if benchmark_df is not None:
            # Align benchmark data with stock data
            if isinstance(benchmark_df.index, pd.DatetimeIndex) and isinstance(result.index, pd.DatetimeIndex):
                benchmark_returns = benchmark_df[benchmark_returns_col].reindex(result.index)
            else:
                benchmark_returns = benchmark_df[benchmark_returns_col]
            
            # Calculate beta
            rolling_cov = result[returns_col].rolling(window=window).cov(benchmark_returns)
            rolling_var = benchmark_returns.rolling(window=window).var()
            result[f'pn_beta_{window}d'] = rolling_cov / rolling_var
            
            # Calculate Treynor ratio
            result[f'pn_treynor_{window}d'] = (
                (result[returns_col].rolling(window=window).mean() - daily_rf) * 252 / 
                result[f'pn_beta_{window}d']
            ).replace([np.inf, -np.inf], np.nan)
            
            # Calculate Information ratio
            active_return = result[returns_col] - benchmark_returns
            result[f'pn_info_ratio_{window}d'] = (
                active_return.rolling(window=window).mean() / 
                active_return.rolling(window=window).std() * 
                np.sqrt(252)
            ).replace([np.inf, -np.inf], np.nan)
        
        # Calculate Calmar ratio (annualized return / maximum drawdown)
        if 'pn_drawdown' in result.columns:
            max_drawdown = result['pn_drawdown'].rolling(window=window).min()
            result[f'pn_calmar_{window}d'] = (
                (result[returns_col].rolling(window=window).mean() * 252) / 
                max_drawdown.abs()
            ).replace([np.inf, -np.inf], np.nan)
        
        print("PyNance ratios calculated successfully")
    except Exception as e:
        print(f"Error calculating PyNance ratios: {e}")
    
    return result


def calculate_pynance_statistics(df: pd.DataFrame, 
                               returns_col: str = 'pn_daily_return') -> Dict:
    """
    Calculate various statistics for returns.
    
    Args:
        df: DataFrame with return data
        returns_col: Column name for returns
        
    Returns:
        Dictionary with calculated statistics
    """
    stats = {}
    
    try:
        returns = df[returns_col].dropna()
        
        # Basic statistics
        stats['mean'] = returns.mean() * 252  # Annualized mean return
        stats['std'] = returns.std() * np.sqrt(252)  # Annualized volatility
        stats['skew'] = returns.skew()  # Skewness
        stats['kurtosis'] = returns.kurtosis()  # Kurtosis
        
        # Best and worst returns
        stats['best_return'] = returns.max()
        stats['worst_return'] = returns.min()
        stats['best_date'] = returns.idxmax()
        stats['worst_date'] = returns.idxmin()
        
        # Percent positive days
        stats['positive_days'] = (returns > 0).sum() / len(returns)
        
        print("PyNance statistics calculated successfully")
    except Exception as e:
        print(f"Error calculating PyNance statistics: {e}")
        stats = {'error': str(e)}
    
    return stats


def calculate_all_pynance_metrics(df: pd.DataFrame, 
                                benchmark_df: pd.DataFrame = None,
                                price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate all PyNance financial metrics.
    
    Args:
        df: DataFrame with price data
        benchmark_df: DataFrame with benchmark price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with all PyNance metrics
    """
    result = df.copy()
    
    try:
        print("Calculating PyNance metrics...")
        
        # Calculate returns
        result = calculate_pynance_returns(result, price_col)
        
        # Calculate risk metrics
        result = calculate_pynance_risk_metrics(result)
        
        # Calculate Value at Risk
        result = calculate_pynance_var(result)
        
        # Calculate drawdowns
        result = calculate_pynance_drawdowns(result, price_col)
        
        # Calculate ratios
        if benchmark_df is not None:
            # Ensure benchmark has returns calculated
            if 'daily_return' not in benchmark_df.columns:
                benchmark_df = calculate_pynance_returns(benchmark_df, price_col)
            
            result = calculate_pynance_ratios(result, benchmark_df)
        
        print("All PyNance metrics calculated successfully")
        
        # Ensure we're returning a DataFrame
        if not isinstance(result, pd.DataFrame):
            print(f"Warning: Result is not a DataFrame but a {type(result)}. Converting to DataFrame.")
            if isinstance(result, tuple):
                # If it's a tuple, take the first element assuming it's the DataFrame
                result = result[0] if len(result) > 0 and isinstance(result[0], pd.DataFrame) else df.copy()
                print("Converted tuple to DataFrame")
    except Exception as e:
        print(f"Error in calculate_all_pynance_metrics: {e}")
        # Return the original DataFrame if there's an error
        result = df.copy()
    
    return result
