import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union

def set_financial_plotting_style():
    """Set the visual style for financial charts."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    sns.set_palette('viridis')

def plot_price_with_moving_averages(df: pd.DataFrame, ticker: str,
                                   ma_columns: List[str] = None,
                                   price_col: str = 'Close',
                                   date_col: str = None) -> plt.Figure:
    """
    Plot stock price with moving averages.
    
    Args:
        df: DataFrame with price and MA data
        ticker: Stock ticker symbol
        ma_columns: List of MA column names to plot
        price_col: Column name for price data
        date_col: Column name for date data (None for DatetimeIndex)
        
    Returns:
        Matplotlib figure
    """
    set_financial_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # If date_col is provided, use it as x-axis, otherwise use index
    x = df[date_col] if date_col is not None else df.index
    
    # Plot price
    ax.plot(x, df[price_col], label=f'{ticker} {price_col}', linewidth=2)
    
    # Plot moving averages
    if ma_columns is None:
        # Try to find MA columns automatically
        ma_columns = [col for col in df.columns if col.startswith('MA_') or col.startswith('EMA_')]
    
    for ma in ma_columns:
        if ma in df.columns:
            ax.plot(x, df[ma], label=ma, alpha=0.7)
    
    ax.set_title(f'{ticker} Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_volume(df: pd.DataFrame, ticker: str,
               volume_col: str = 'Volume',
               date_col: str = None) -> plt.Figure:
    """
    Plot trading volume.
    
    Args:
        df: DataFrame with volume data
        ticker: Stock ticker symbol
        volume_col: Column name for volume data
        date_col: Column name for date data (None for DatetimeIndex)
        
    Returns:
        Matplotlib figure
    """
    set_financial_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # If date_col is provided, use it as x-axis, otherwise use index
    x = df[date_col] if date_col is not None else df.index
    
    # Plot volume bars
    ax.bar(x, df[volume_col], alpha=0.5, color='navy')
    
    ax.set_title(f'{ticker} Trading Volume')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1e6)}M'))
    
    plt.tight_layout()
    return fig

def plot_technical_indicators(df: pd.DataFrame, ticker: str,
                            indicators: Dict[str, List[str]],
                            date_col: str = None) -> Dict[str, plt.Figure]:
    """
    Plot multiple technical indicators.
    
    Args:
        df: DataFrame with technical indicator data
        ticker: Stock ticker symbol
        indicators: Dictionary mapping indicator types to column names
        date_col: Column name for date data (None for DatetimeIndex)
        
    Returns:
        Dictionary mapping indicator types to Matplotlib figures
    """
    set_financial_plotting_style()
    figures = {}
    
    # If date_col is provided, use it as x-axis, otherwise use index
    x = df[date_col] if date_col is not None else df.index
    
    # Plot each group of indicators
    for indicator_type, columns in indicators.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in columns:
            if col in df.columns:
                ax.plot(x, df[col], label=col)
        
        ax.set_title(f'{ticker} {indicator_type}')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures[indicator_type] = fig
    
    return figures

def plot_candlestick(df: pd.DataFrame, ticker: str, ax=None, 
                    date_col: str = None) -> plt.Figure:
    """
    Plot candlestick chart.
    
    Args:
        df: DataFrame with OHLC data
        ticker: Stock ticker symbol
        ax: Matplotlib axis to plot on (if None, creates new figure)
        date_col: Column name for date data (None for DatetimeIndex)
        
    Returns:
        Matplotlib figure
    """
    set_financial_plotting_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # If date_col is provided, use it as x-axis, otherwise use index
    x = df[date_col] if date_col is not None else df.index
    
    # Create a numeric x-axis for bar plotting
    x_numeric = np.arange(len(x))
    
    # Width of bars
    width = 0.6
    
    # Up and down days
    up = df['Close'] > df['Open']
    down = df['Open'] > df['Close']
    
    # Plot candlesticks
    # Bodies
    ax.bar(x_numeric[up], df['Close'][up] - df['Open'][up], width, bottom=df['Open'][up], color='green', alpha=0.7)
    ax.bar(x_numeric[down], df['Open'][down] - df['Close'][down], width, bottom=df['Close'][down], color='red', alpha=0.7)
    
    # Wicks
    ax.vlines(x_numeric[up], df['Low'][up], df['High'][up], color='green', linewidth=1)
    ax.vlines(x_numeric[down], df['Low'][down], df['High'][down], color='red', linewidth=1)
    
    # Set x-axis ticks
    step = max(1, len(x) // 10)  # Show ~10 dates on x-axis
    ax.set_xticks(x_numeric[::step])
    ax.set_xticklabels(x[::step], rotation=45)
    
    ax.set_title(f'{ticker} Candlestick Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_combined_analysis(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Create a comprehensive plot with price, volume, and key indicators.
    
    Args:
        df: DataFrame with price, volume, and indicator data
        ticker: Stock ticker symbol
        
    Returns:
        Matplotlib figure
    """
    set_financial_plotting_style()
    
    # Create subplot grid
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # Price and MAs
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    
    # Plot MAs if available
    for ma in [col for col in df.columns if col.startswith('MA_') or col.startswith('EMA_')][:3]:
        if ma in df.columns:
            ax1.plot(df.index, df[ma], label=ma, alpha=0.7)
    
    # Plot Bollinger Bands if available
    if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
        ax1.fill_between(df.index, df['BBL_20_2.0'], df['BBU_20_2.0'], 
                        color='gray', alpha=0.2, label='Bollinger Bands')
    
    ax1.set_title(f'{ticker} Price Analysis')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(df.index, df['Volume'], color='navy', alpha=0.5)
    ax2.set_ylabel('Volume')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1e6)}M'))
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if 'RSI_14' in df.columns:
        ax3.plot(df.index, df['RSI_14'], color='purple')
        ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    # MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
        ax4.plot(df.index, df['MACD_12_26_9'], color='blue', label='MACD')
        ax4.plot(df.index, df['MACDs_12_26_9'], color='red', label='Signal')
        
        # Calculate and plot histogram
        hist = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        ax4.bar(df.index, hist, color='gray', alpha=0.5, label='Histogram')
        
        ax4.set_ylabel('MACD')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Date')
    
    # Format x-axis dates for bottom subplot only
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(stocks_data: Dict[str, pd.DataFrame], 
                          returns_col: str = 'daily_return') -> plt.Figure:
    """
    Plot correlation matrix of stock returns.
    
    Args:
        stocks_data: Dictionary mapping tickers to DataFrames with return data
        returns_col: Column name for returns
        
    Returns:
        Matplotlib figure
    """
    set_financial_plotting_style()
    
    # Extract returns from each stock
    returns = {}
    for ticker, df in stocks_data.items():
        if returns_col in df.columns:
            returns[ticker] = df[returns_col]
    
    if not returns:
        return None
    
    # Create DataFrame with all returns
    returns_df = pd.DataFrame(returns)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Correlation Matrix of Stock Returns')
    
    plt.tight_layout()
    return fig

def plot_returns_distribution(df: pd.DataFrame, ticker: str,
                           returns_col: str = 'daily_return') -> plt.Figure:
    """
    Plot distribution of returns.
    
    Args:
        df: DataFrame with return data
        ticker: Stock ticker symbol
        returns_col: Column name for returns
        
    Returns:
        Matplotlib figure
    """
    set_financial_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram of returns
    sns.histplot(df[returns_col].dropna(), kde=True, ax=ax)
    
    # Add vertical line at 0
    ax.axvline(0, color='red', linestyle='--')
    
    # Add normal distribution fit
    mu = df[returns_col].mean()
    sigma = df[returns_col].std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
    # Scale y to match histogram
    hist_max = ax.get_ylim()[1]
    pdf_max = max(y)
    y = y * (hist_max / pdf_max)
    ax.plot(x, y, 'r-', label='Normal Distribution')
    
    ax.set_title(f'{ticker} Returns Distribution')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    return fig
