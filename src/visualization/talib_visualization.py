import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Optional, Tuple
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# --------------------------
# Style Settings
# --------------------------
def set_style():
    """Set the default style for financial plots."""
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

# --------------------------
# Moving Averages
# --------------------------
def add_moving_averages(df: pd.DataFrame, ma_periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
    """Add SMA and EMA for given periods."""
    df = df.copy()
    for period in ma_periods:
        df[f'ta_sma_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'ta_ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

# --------------------------
# Price and Moving Averages
# --------------------------
def plot_price_with_ma(df: pd.DataFrame, ticker: str,
                       ma_types: List[str] = ['ta_sma', 'ta_ema'],
                       ma_periods: List[int] = [20, 50, 200],
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    # Plot Close price
    ax.plot(df['Date'], df['Close'], label=f'{ticker} Close', color='black', linewidth=1.5)

    # Plot MAs
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    color_idx = 0
    for ma_type in ma_types:
        for period in ma_periods:
            col_name = f'{ma_type}_{period}'
            if col_name in df.columns:
                ax.plot(df['Date'], df[col_name], 
                        label=f'{ma_type.replace("ta_", "").upper()} {period}',
                        color=colors[color_idx % len(colors)],
                        linewidth=1.0, alpha=0.8)
                color_idx += 1

    # Formatting
    ax.set_title(f'{ticker} Price with Moving Averages', fontsize=14)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.grid(True)
    return fig

# --------------------------
# Momentum Indicators
# --------------------------
def plot_momentum_indicators(df: pd.DataFrame, ticker: str,
                             indicators: List[str] = ['ta_rsi', 'ta_stoch_k', 'ta_stoch_d'],
                             ax: Optional[plt.Axes] = None) -> plt.Figure:
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, indicator in enumerate(indicators):
        if indicator in df.columns:
            ax.plot(df.index, df[indicator], label=indicator.replace('ta_', '').upper(),
                    color=colors[i % len(colors)])

    # Reference lines
    if 'ta_rsi' in indicators:
        ax.axhline(70, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(30, color='gray', linestyle='--', alpha=0.5)
    if 'ta_stoch_k' in indicators or 'ta_stoch_d' in indicators:
        ax.axhline(80, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(20, color='gray', linestyle='--', alpha=0.5)

    ax.set_title(f'{ticker} Momentum Indicators', fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    return fig

# --------------------------
# Volatility Indicators
# --------------------------
def plot_volatility_indicators(df: pd.DataFrame, ticker: str,
                               ax: Optional[plt.Axes] = None) -> plt.Figure:
    set_style()
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axes
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = ax.twinx()

    # Price + Bollinger Bands
    ax1.plot(df.index, df['Close'], color='black', label='Close', linewidth=1.5)
    if 'ta_bollinger_hband' in df.columns:
        ax1.plot(df.index, df['ta_bollinger_hband'], label='Upper BB', color='red', alpha=0.7)
        ax1.plot(df.index, df['ta_bollinger_mavg'], label='Middle BB', color='blue', alpha=0.7)
        ax1.plot(df.index, df['ta_bollinger_lband'], label='Lower BB', color='red', alpha=0.7)
        ax1.fill_between(df.index, df['ta_bollinger_hband'], df['ta_bollinger_lband'], color='gray', alpha=0.2)

    # ATR
    if 'ta_atr' in df.columns and isinstance(ax2, plt.Axes):
        ax2.plot(df.index, df['ta_atr'], label='ATR', color='purple', linewidth=1.0)
        ax2.set_ylabel('ATR', fontsize=12)
        ax2.legend(loc='upper right')

    # Formatting
    ax1.set_title(f'{ticker} Volatility Indicators', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# --------------------------
# MACD
# --------------------------
def plot_macd(df: pd.DataFrame, ticker: str,
              ax: Optional[plt.Axes] = None) -> plt.Figure:
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    if all(col in df.columns for col in ['ta_macd', 'ta_macd_signal', 'ta_macd_diff']):
        ax.plot(df.index, df['ta_macd'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(df.index, df['ta_macd_signal'], label='Signal', color='red', linewidth=1.5)
        for i in range(len(df) - 1):
            color = 'green' if df['ta_macd_diff'].iloc[i] >= 0 else 'red'
            ax.bar(df.index[i], df['ta_macd_diff'].iloc[i], color=color, width=0.7, alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{ticker} MACD', fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    return fig

# --------------------------
# Trading Signals
# --------------------------
def plot_trading_signals(df: pd.DataFrame, ticker: str,
                         price_col: str = 'Close',
                         buy_col: str = 'buy_signal',
                         sell_col: str = 'sell_signal',
                         ax: Optional[plt.Axes] = None) -> plt.Figure:
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    ax.plot(df.index, df[price_col], color='black', label=f'{ticker} {price_col}', linewidth=1.5)
    if buy_col in df.columns:
        ax.scatter(df.index[df[buy_col] == 1], df[price_col][df[buy_col] == 1],
                   marker='^', color='green', s=100, label='Buy Signal')
    if sell_col in df.columns:
        ax.scatter(df.index[df[sell_col] == 1], df[price_col][df[sell_col] == 1],
                   marker='v', color='red', s=100, label='Sell Signal')

    ax.set_title(f'{ticker} Trading Signals', fontsize=14)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    return fig

# --------------------------
# Strategy Performance
# --------------------------
def plot_strategy_performance(df: pd.DataFrame, ticker: str,
                              ax: Optional[plt.Axes] = None) -> plt.Figure:
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    if 'strategy_cumulative_return' in df.columns and 'buy_hold_cumulative_return' in df.columns:
        ax.plot(df.index, df['strategy_cumulative_return'] * 100, label='Strategy', color='blue', linewidth=2.0)
        ax.plot(df.index, df['buy_hold_cumulative_return'] * 100, label='Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{ticker} Strategy Performance', fontsize=14)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    return fig

# --------------------------
# PyNance Metrics
# --------------------------
def plot_pynance_metrics(df: pd.DataFrame, ticker: str,
                         metrics: List[str] = ['pn_volatility_252d', 'pn_sharpe_252d'],
                         ax: Optional[plt.Axes] = None) -> plt.Figure:
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.plot(df.index, df[metric],
                    label=metric.replace('pn_', '').replace('_', ' ').title(),
                    color=colors[i % len(colors)])
    ax.set_title(f'{ticker} PyNance Metrics', fontsize=14)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    return fig

# --------------------------
# Correlation Matrix
# --------------------------
def plot_correlation_matrix(df: pd.DataFrame, title: str = 'Correlation Matrix',
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    set_style()
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig

# --------------------------
# Full Dashboard
# --------------------------
def create_full_analysis_dashboard(df: pd.DataFrame, ticker: str) -> plt.Figure:
    set_style()
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[3, 2, 2, 2, 2])
    plot_price_with_ma(df, ticker, ax=fig.add_subplot(gs[0, :]))
    plot_momentum_indicators(df, ticker, ax=fig.add_subplot(gs[1, 0]))
    plot_macd(df, ticker, ax=fig.add_subplot(gs[1, 1]))
    plot_volatility_indicators(df, ticker, ax=fig.add_subplot(gs[2, :]))
    plot_trading_signals(df, ticker, ax=fig.add_subplot(gs[3, 0]))
    plot_strategy_performance(df, ticker, ax=fig.add_subplot(gs[3, 1]))
    plot_pynance_metrics(df, ticker, ax=fig.add_subplot(gs[4, :]))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    return fig
