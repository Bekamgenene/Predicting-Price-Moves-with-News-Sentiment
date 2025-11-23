import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# Set default style for plots
def set_style():
    """Set the default style for financial plots."""
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_price_with_ma(df: pd.DataFrame, 
                     ticker: str, 
                     ma_types: List[str] = ['ta_sma', 'ta_ema'],
                     ma_periods: List[int] = [20, 50, 200],
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot price with moving averages.
    
    Args:
        df: DataFrame with price and MA data
        ticker: Ticker symbol
        ma_types: List of MA types to include
        ma_periods: List of MA periods to include
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Plot price
    ax.plot(df.index, df['Close'], label=f'{ticker} Close', color='black', linewidth=1.5)
    
    # Plot moving averages
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    color_idx = 0
    
    for ma_type in ma_types:
        for period in ma_periods:
            col_name = f'{ma_type}_{period}'
            if col_name in df.columns:
                ax.plot(df.index, df[col_name], 
                        label=f'{ma_type.replace("ta_", "").upper()} {period}', 
                        color=colors[color_idx % len(colors)], 
                        linewidth=1.0, 
                        alpha=0.8)
                color_idx += 1
    
    # Format plot
    ax.set_title(f'{ticker} Price with Moving Averages', fontsize=14)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    return fig

def plot_momentum_indicators(df: pd.DataFrame, 
                           ticker: str,
                           indicators: List[str] = ['ta_rsi', 'ta_stoch_k', 'ta_stoch_d'],
                           ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot momentum indicators.
    
    Args:
        df: DataFrame with indicator data
        ticker: Ticker symbol
        indicators: List of indicators to include
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot each indicator
    for i, indicator in enumerate(indicators):
        if indicator in df.columns:
            ax.plot(df.index, df[indicator], 
                    label=indicator.replace('ta_', '').upper(), 
                    color=colors[i % len(colors)])
    
    # Add reference lines
    if 'ta_rsi' in indicators:
        ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
    
    if 'ta_stoch_k' in indicators or 'ta_stoch_d' in indicators:
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
    
    # Format plot
    ax.set_title(f'{ticker} Momentum Indicators', fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    return fig

def plot_volatility_indicators(df: pd.DataFrame, 
                             ticker: str,
                             ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot volatility indicators (Bollinger Bands, ATR).
    
    Args:
        df: DataFrame with indicator data
        ticker: Ticker symbol
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axes
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = ax.twinx()
    
    # Plot price and Bollinger Bands on first axis
    ax1.plot(df.index, df['Close'], label='Close', color='black', linewidth=1.5)
    
    if 'ta_bollinger_hband' in df.columns:
        ax1.plot(df.index, df['ta_bollinger_hband'], label='Upper BB', color='red', alpha=0.7)
        ax1.plot(df.index, df['ta_bollinger_mavg'], label='Middle BB', color='blue', alpha=0.7)
        ax1.plot(df.index, df['ta_bollinger_lband'], label='Lower BB', color='red', alpha=0.7)
        
        # Fill between bands
        ax1.fill_between(df.index, df['ta_bollinger_hband'], df['ta_bollinger_lband'], 
                        color='gray', alpha=0.2)
    
    # Plot ATR on second axis
    if 'ta_atr' in df.columns and isinstance(ax2, plt.Axes):
        ax2.plot(df.index, df['ta_atr'], label='ATR', color='purple', linewidth=1.0)
        ax2.set_ylabel('ATR', fontsize=12)
        ax2.legend(loc='upper right')
    
    # Format plots
    ax1.set_title(f'{ticker} Volatility Indicators', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_macd(df: pd.DataFrame, 
            ticker: str,
            ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot MACD indicator.
    
    Args:
        df: DataFrame with MACD data
        ticker: Ticker symbol
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    if 'ta_macd' in df.columns and 'ta_macd_signal' in df.columns and 'ta_macd_diff' in df.columns:
        # Plot MACD line and signal line
        ax.plot(df.index, df['ta_macd'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(df.index, df['ta_macd_signal'], label='Signal', color='red', linewidth=1.5)
        
        # Plot histogram
        for i in range(len(df) - 1):
            if df['ta_macd_diff'].iloc[i] >= 0:
                ax.bar(df.index[i], df['ta_macd_diff'].iloc[i], color='green', width=0.7, alpha=0.5)
            else:
                ax.bar(df.index[i], df['ta_macd_diff'].iloc[i], color='red', width=0.7, alpha=0.5)
        
        # Zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Format plot
        ax.set_title(f'{ticker} MACD', fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    
    return fig

def plot_trading_signals(df: pd.DataFrame, 
                       ticker: str,
                       price_col: str = 'Close',
                       buy_col: str = 'buy_signal',
                       sell_col: str = 'sell_signal',
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot price with buy and sell signals.
    
    Args:
        df: DataFrame with price and signal data
        ticker: Ticker symbol
        price_col: Column name for price
        buy_col: Column name for buy signals
        sell_col: Column name for sell signals
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Plot price
    ax.plot(df.index, df[price_col], label=f'{ticker} {price_col}', color='black', linewidth=1.5)
    
    # Plot buy signals
    if buy_col in df.columns:
        buy_signals = df[df[buy_col] == 1]
        ax.scatter(buy_signals.index, buy_signals[price_col], 
                  marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    if sell_col in df.columns:
        sell_signals = df[df[sell_col] == 1]
        ax.scatter(sell_signals.index, sell_signals[price_col], 
                  marker='v', color='red', s=100, label='Sell Signal')
    
    # Format plot
    ax.set_title(f'{ticker} Trading Signals', fontsize=14)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    return fig

def plot_strategy_performance(df: pd.DataFrame,
                            ticker: str,
                            ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot strategy performance compared to buy and hold.
    
    Args:
        df: DataFrame with strategy performance data
        ticker: Ticker symbol
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Plot strategy and buy-hold cumulative returns
    if 'strategy_cumulative_return' in df.columns and 'buy_hold_cumulative_return' in df.columns:
        ax.plot(df.index, df['strategy_cumulative_return'] * 100, 
                label='Strategy', color='blue', linewidth=2.0)
        ax.plot(df.index, df['buy_hold_cumulative_return'] * 100, 
                label='Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
        
        # Format as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Format plot
        ax.set_title(f'{ticker} Strategy Performance', fontsize=14)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    return fig

def plot_pynance_metrics(df: pd.DataFrame,
                       ticker: str,
                       metrics: List[str] = ['pn_volatility_252d', 'pn_sharpe_252d'],
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot PyNance metrics.
    
    Args:
        df: DataFrame with PyNance metrics
        ticker: Ticker symbol
        metrics: List of metrics to plot
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.plot(df.index, df[metric], 
                    label=metric.replace('pn_', '').replace('_', ' ').title(), 
                    color=colors[i % len(colors)])
    
    # Format plot
    ax.set_title(f'{ticker} PyNance Metrics', fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    return fig

def plot_correlation_matrix(df: pd.DataFrame,
                          title: str = 'Correlation Matrix',
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot correlation matrix for financial indicators.
    
    Args:
        df: DataFrame with indicators to correlate
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Format plot
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    
    return fig

def create_full_analysis_dashboard(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Create a comprehensive dashboard with multiple indicators and metrics.
    
    Args:
        df: DataFrame with price, indicators, and metrics data
        ticker: Ticker symbol
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[3, 2, 2, 2, 2])
    
    # Plot price with MAs
    ax1 = fig.add_subplot(gs[0, :])
    plot_price_with_ma(df, ticker, ax=ax1)
    
    # Plot momentum indicators
    ax2 = fig.add_subplot(gs[1, 0])
    plot_momentum_indicators(df, ticker, ax=ax2)
    
    # Plot MACD
    ax3 = fig.add_subplot(gs[1, 1])
    plot_macd(df, ticker, ax=ax3)
    
    # Plot volatility indicators
    ax4 = fig.add_subplot(gs[2, :])
    plot_volatility_indicators(df, ticker, ax=ax4)
    
    # Plot trading signals
    ax5 = fig.add_subplot(gs[3, 0])
    plot_trading_signals(df, ticker, ax=ax5)
    
    # Plot strategy performance
    ax6 = fig.add_subplot(gs[3, 1])
    plot_strategy_performance(df, ticker, ax=ax6)
    
    # Plot PyNance metrics
    ax7 = fig.add_subplot(gs[4, :])
    plot_pynance_metrics(df, ticker, ax=ax7)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    return fig
